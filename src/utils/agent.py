import torch
from torch_ac.belief import threshold_rm_beliefs
import torch.nn.functional as F
import utils
from model import ACModel, RecurrentACModel
from detector_model import getDetectorModel

class Agent:
    def __init__(self, env, obs_space, action_space, model_dir,
                rm_update_algo, hidden_size=128, use_mem=False, use_mem_detector=False,
                device=None):
        try:
            print(model_dir)
            status = utils.get_status(model_dir)
        except OSError:
            status = {"num_frames": 0, "update": 0}

        self.env = env
        self.rm_update_algo = rm_update_algo
        self.use_mem = use_mem
        self.use_rm_belief = (rm_update_algo in ["tdm", "naive", "ibu"]) 

        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(env)  

        self.detectormodel = getDetectorModel(env, obs_space, rm_update_algo, use_mem_detector)
        if use_mem_detector:
            self.detector_memories = torch.zeros(1, self.detectormodel.memory_size, device=device)

        if use_mem:
            acmodel = RecurrentACModel(env, obs_space, env.action_space, rm_update_algo, hidden_size=hidden_size)
            self.memories = torch.zeros(1, acmodel.memory_size, device=device)
        else:
            acmodel = ACModel(env, obs_space, env.action_space, rm_update_algo, hidden_size=hidden_size)

        self.acmodel = acmodel
        print(acmodel)
        self.device = device

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(self.device)
        self.acmodel.eval()

        self.detectormodel.load_state_dict(utils.get_detector_model_state(model_dir))
        self.detectormodel.to(self.device)
        self.detectormodel.eval()

    def get_action(self, obs):
        preprocessed_obs = self.preprocess_obss([obs], device=self.device)

        # Perform RM updates...
        with torch.no_grad():
            # Generate a detector belief
            if self.rm_update_algo in ["tdm", "naive", "ibu"]:
                if self.detectormodel.recurrent:
                    detector_belief, self.detector_memories = self.detectormodel(preprocessed_obs, self.detector_memories)
                else:
                    detector_belief = self.detectormodel(preprocessed_obs)
            
            if self.rm_update_algo == "tdm":
                detector_belief = F.softmax(detector_belief)

            if self.rm_update_algo == "naive":
                detector_belief = self.env.update_rm_beliefs((detector_belief > 0).cpu().numpy()[0, :])  # diff from collect_experience()
                detector_belief = torch.tensor(detector_belief, device=self.device, dtype=torch.float).unsqueeze(0)

            if self.rm_update_algo == "ibu":
                detector_belief = self.env.update_rm_beliefs(detector_belief.cpu().numpy()[0, :])  # diff from collect_experience()
                detector_belief = torch.tensor(detector_belief, device=self.device, dtype=torch.float).unsqueeze(0)

            ## If necessary, add rm_belief to the observation
            if self.rm_update_algo in ["tdm", "naive", "ibu"]:
                preprocessed_obs.rm_belief = detector_belief
                self.rm_belief = detector_belief.squeeze()
                
            # Get policy action
            if self.acmodel.recurrent:
                dist, value, self.memories = self.acmodel(preprocessed_obs, self.memories)
            else:
                dist, value = self.acmodel(preprocessed_obs)

            return dist.sample().cpu().numpy()[0]

    def analyze_feedback(self, done):
        masks = 1 - torch.tensor([done], dtype=torch.float).unsqueeze(1)

        if self.acmodel.recurrent:
            self.memories *= masks
        if self.detectormodel.recurrent:
            self.detector_memories *= masks

    def pretty_print_belief(self):
        belief_str = []
        if self.rm_update_algo in ["tdm", "ibu", "naive"]:
            for k in self.rm_belief:
                belief_str.append("%.2f"%(k))

        return "[" + ", ".join(belief_str) + "]"
