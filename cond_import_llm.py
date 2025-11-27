try:
    import lmstudio as lms
    lms.get_default_client(SERVER_API_HOST)

    def get_llm_response(input_text):
        with lms.Client() as client:
            model = client.llm.model("") # Helper model
            llm_response = model.respond(input_text)

        llm_response = str(llm_response)
        return llm_response

except ImportError:
    from openai import OpenAI
    client = OpenAI(base_url=SERVER_API_HOST, api_key="lm-studio")

    def get_llm_response(input_text):
        response_obj = client.chat.completions.create(
            model="",
            messages=[{"role": "user", "content": input_text}],
            temperature=0.7,
            max_tokens=100
        )
        return response_obj.choices[0].message.content
