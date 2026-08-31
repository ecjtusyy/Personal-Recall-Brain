from second_brain.agent.lfm_adapter import OpenVINOLFMModel


class FakePipeline:
    def __init__(self):
        self.kwargs = None

    def get_tokenizer(self):
        return self

    def apply_chat_template(self, messages, add_generation_prompt):
        assert messages == [{"role": "user", "content": "测试"}]
        assert add_generation_prompt
        return "FORMATTED<think>"

    def generate(self, prompt, **kwargs):
        assert prompt == (
            "FORMATTED<think>I will follow the instructions and return only the requested result."
            "</think>\n"
        )
        self.kwargs = kwargs
        return "本地模型可用"


def test_lfm_uses_official_low_temperature_generation_settings(tmp_path):
    model = OpenVINOLFMModel(tmp_path)
    model._pipeline = FakePipeline()
    assert model.loaded
    assert model.generate("测试", 80) == "本地模型可用"
    assert model._pipeline.kwargs == {
        "max_new_tokens": 80,
        "apply_chat_template": False,
        "do_sample": True,
        "temperature": 0.1,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "rng_seed": 42,
    }


def test_lfm_removes_unexpected_reasoning_prefix(tmp_path):
    model = OpenVINOLFMModel(tmp_path)
    model._pipeline = FakePipeline()
    model._pipeline.generate = lambda prompt, **kwargs: "额外推理</think>最终答案"
    assert model.generate("测试", 5) == "最终答案"
