from second_brain.agent.lfm_adapter import OpenVINOLFMModel


class FakePipeline:
    def __init__(self):
        self.kwargs = None

    def generate(self, prompt, **kwargs):
        self.kwargs = kwargs
        return "本地模型可用"


def test_lfm_uses_official_low_temperature_generation_settings(tmp_path):
    model = OpenVINOLFMModel(tmp_path)
    model._pipeline = FakePipeline()
    assert model.loaded
    assert model.generate("测试", 80) == "本地模型可用"
    assert model._pipeline.kwargs == {
        "max_new_tokens": 80,
        "do_sample": True,
        "temperature": 0.1,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "rng_seed": 42,
    }
