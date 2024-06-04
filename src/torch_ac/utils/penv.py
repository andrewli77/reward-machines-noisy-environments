from multiprocessing import Process, Pipe
import gymnasium

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            if terminated or truncated:
                obs = env.reset()
            conn.send((obs, reward, terminated, truncated, info))
        elif cmd == "reset":
            obs, _ = env.reset()
            conn.send(obs)
        elif cmd == "update_rm_beliefs":
            belief_u = env.update_rm_beliefs(data)
            conn.send(belief_u)
        elif cmd == "kill":
            return
        else:
            raise NotImplementedError

class ParallelEnv(gymnasium.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def __del__(self):
        for local in self.locals:
            local.send(("kill", None))

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()[0]] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, terminated, truncated, info = self.envs[0].step(actions[0])
        if terminated or truncated:
            obs, _ = self.envs[0].reset()
        results = zip(*[(obs, reward, terminated, truncated, info)] + [local.recv() for local in self.locals])
        return results

    def update_rm_beliefs(self, event_preds):
        for local, eps in zip(self.locals, event_preds[1:]):
            local.send(("update_rm_beliefs", eps))
        results = [self.envs[0].update_rm_beliefs(event_preds[0])] + [local.recv() for local in self.locals]
        return results

    def render(self):
        raise NotImplementedError