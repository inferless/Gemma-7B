from vllm import LLM, SamplingParams

class InferlessPythonModel:
    def initialize(self):
        repo_id = "google/gemma-7b"
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95,max_tokens=256)
        self.llm = LLM(model=repo_id)

    def infer(self, inputs):
        prompts = inputs["prompt"]
        result = self.llm.generate(prompts, self.sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {'gresult': result_output[0]}

    def finalize(self):
        pass
