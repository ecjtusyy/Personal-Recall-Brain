from second_brain.agent.model_orchestrator import ModelOrchestrator


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.available = True
        self.loaded = False
        self.events = []

    def warmup(self):
        self.loaded = True
        self.events.append("warmup")

    def generate(self, prompt, max_new_tokens=None, image_paths=None):
        self.loaded = True
        self.events.append(("generate", prompt))
        return self.name

    def unload(self):
        self.loaded = False
        self.events.append("unload")


def test_deep_recall_never_keeps_two_models_loaded():
    controller = FakeModel("lfm")
    reasoner = FakeModel("qwen")
    runtime = ModelOrchestrator(controller, reasoner)
    runtime.warmup()
    assert controller.loaded and not reasoner.loaded

    result = runtime.analyze_deep("分析轨迹")

    assert result == "qwen"
    assert controller.loaded and not reasoner.loaded
    assert controller.events[-2:] == ["unload", "warmup"]
    assert reasoner.events[-2:] == [("generate", "分析轨迹"), "unload"]


def test_fast_recall_uses_resident_controller():
    controller = FakeModel("lfm")
    reasoner = FakeModel("qwen")
    runtime = ModelOrchestrator(controller, reasoner)
    runtime.warmup()
    assert runtime.generate("规划") == "lfm"
    assert reasoner.events == ["unload"]
