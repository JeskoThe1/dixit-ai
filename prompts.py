from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.llms import OpenAI
import numpy as np


def get_image_interpretation_chain(model='gpt-3.5-turbo-instruct', verbose=True):
    desc_llm = OpenAI(model_name=model, max_tokens=512)
    desc_prompt = PromptTemplate(
        input_variables=["image_descriptions", "ai_models"],
        template=(
            "I give you a list of descriptions of the same image by "
            "different AI models ({ai_models}), where BLIP-2 is the most trust-worthy. "
            "Here is the list of descriptions: "
            "\n"
            "{image_descriptions}"
            "\n"
            "Can you describe in detail how do you imagine this image looks "
            "like? What characters and objects are on the image, and what they are doing? "
            "Be specific. By the way, it might be not an image of the real world."))
    desc_chain = LLMChain(
        llm=desc_llm, prompt=desc_prompt,
        output_key="image_interpretation", verbose=verbose)
    return desc_chain


def talk_with_blip2(image_interpretation,
                    image,
                    ask_blip2_fn,
                    clue=None,
                    num_questions=2,
                    model='gpt-3.5-turbo-instruct',
                    verbose=True):
    question_answering_log = []
    blip2_answer = "Only ask me questions that matters."
    if clue is not None:
        # Can we directly ask BLIP2 to explain the connection? believe in T5!
        direct_question = f"How does this image relate to the phrase {clue}"
        question_answering_log.append("Question: " + direct_question)
        blip2_answer = ask_blip2_fn(image, direct_question)
        if verbose:
            print("Answer:", blip2_answer)
        question_answering_log.append("Answer: " + blip2_answer.strip())
    
    # Think about what we want to ask
    image_interpretation = image_interpretation.strip()

    pre_llm = OpenAI(model_name=model, max_tokens=256)
    pre_prompt = PromptTemplate(
        input_variables=["image_interpretation"],
        template=(
            "You can't see this photo but you are given its description by AI models "
            "(the most trust-worthy is BLIP-2):"
            "\n"
            "{image_interpretation}"
            "\n"
            "What additional information do you need to be able to tell "
            "a compelling story about what is happening in this photo?"))
    pre_chain = LLMChain(llm=pre_llm, prompt=pre_prompt, verbose=verbose)
    pre_results = pre_chain.predict(image_interpretation=image_interpretation)
    pre_results = pre_results.strip()

    llm = OpenAI(model_name=model, max_tokens=512)
    prompt = PromptTemplate(
        input_variables=["blip2_answer", "chat_history"],
        template=(
            "Your name is Bob, you're talking with Alice to understand "
            "a photo. Your task is to get more information "
            "about the photo from Alice by asking her short questions. "
            "Hint - a good question is about the actions happening in the "
            "photo and what a specific character is doing, or about other "
            "objects or living creatures that are present in the photo."
            "\n"
            f"{image_interpretation.strip()}"
            "\n"
            f"{pre_results}"
            "\n"
            "{chat_history}"
            "\n"
            "Alice: {blip2_answer}"
            "\n"
            "Alice: You can ask me one short question about the photo. "
            "\n"
            "Bob:"))
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="Alice",
        ai_prefix="Bob")
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=verbose)

    for iter in range(num_questions):
        results = chain.predict(
            blip2_answer=blip2_answer.strip(),
        )
        if verbose:
            print('Question:', results)
        
        # Sometimes, the result can contain hallucinated answer so we should
        # filter it out:
        results = results[:results.find("Alice:")]
        question_answering_log.append("Question: " + results.strip())
        blip2_answer = ask_blip2_fn(image, results.strip())
        if verbose:
            print("Answer:", blip2_answer)
        question_answering_log.append("Answer: " + blip2_answer.strip())

    return '\n'.join(question_answering_log)


def get_post_qna_inpterpretation_chain(model='gpt-3.5-turbo-instruct', verbose=True):
    desc_llm = OpenAI(model_name=model, max_tokens=512)
    desc_prompt = PromptTemplate(
        input_variables=[
            "captions", "qna_session", "ai_models"],
        template=(
            "Your task is to generate a detailed description of an image "
            "that you don't see. Here is the descriptions generated by "
            "AI models ({ai_models}), they can be inaccurate:"
            "\n"
            "{captions}"
            "\n"
            "Here is the QnA session with BLIP-2, an AI model that can answer "
            "question about the image (may be inaccurate as well):"
            "\n"
            "{qna_session}"
            "\n"
            "Can you describe in detail how do you imagine this image looks "
            "like? What characters and objects are on the image, and what they are doing? "
            "Be specific. By the way, it might be not an image of the real world."))
    desc_chain = LLMChain(
        llm=desc_llm, prompt=desc_prompt,
        output_key="image_interpretation", verbose=verbose)
    return desc_chain


def get_clue_chain(model='gpt-3.5-turbo-instruct', verbose=True):
    association_llm = OpenAI(model_name=model, max_tokens=512)
    association_prompt = PromptTemplate(
        input_variables=["image_interpretation"],
        template=(
            "For an image with a following detailed description:"
            "\n"
            "{image_interpretation}"
            "\n"
            "What does it associate with for you? Be abstract and creative! "
            "Any phylosophical thoughts? What does it remind you about?"))
    association_chain = LLMChain(
        llm=association_llm, prompt=association_prompt,
        output_key="association", verbose=verbose)

    clue_llm = OpenAI(model_name=model, max_tokens=512)
    clue_prompt = PromptTemplate(
        input_variables=["association", "personality"],
        template=(
            "Given the following association for an image"
            "\n"
            "{association}"
            "\n"
            "Considering the provided personality traits : {personality}, summarize the specified association. "
            "Pay attention to how the personality influences the dynamics, goals, and overall atmosphere of the association. "
            "Summarize association in one short phrase, no more than 3 words. More than three words is a violation of the rules"
        ))
    clue_chain = LLMChain(
        llm=clue_llm, prompt=clue_prompt,
        output_key="clue", verbose=verbose)
    return SequentialChain(
        chains=[association_chain, clue_chain],
        input_variables=["image_interpretation", "personality"],
        output_variables=["clue", "association"],
        verbose=verbose)


def generate_clue_for_image(image,
                            generate_captions_fn,
                            models,
                            personality='generic',
                            openai_model='gpt-3.5-turbo-instruct',
                            num_blip2_questions=3,
                            verbose=True):
    
    captioning_results = generate_captions_fn(image, models)
    
    pre_qna_interpretation = ""
    image_interpretation = ""
    blip2_results = ""
    
    # Get first interpretation
    image_interp_chain = get_image_interpretation_chain(
        model=openai_model, verbose=verbose)
    image_interpretation = image_interp_chain.predict(
        image_descriptions=captioning_results["captions"],
        ai_models=", ".join(captioning_results["models"]))
    image_interpretation = image_interpretation.strip()
    pre_qna_interpretation = image_interpretation

    if num_blip2_questions > 0:
        # Talk with BLIP-v2 to get more information
        blip2_results = talk_with_blip2(
            image_interpretation=image_interpretation,
            image=image,
            ask_blip2_fn=models.blip2,
            num_questions=num_blip2_questions,
            model=openai_model,
            verbose=verbose)
        blip2_results = blip2_results.strip()

        # Get final interpretation after QnA session:
        final_interp_chain = get_post_qna_inpterpretation_chain(
            model=openai_model, verbose=verbose)
        image_interpretation = final_interp_chain.predict(
            captions=captioning_results["captions"],
            ai_models=", ".join(captioning_results["models"]),
            qna_session=blip2_results)
        image_interpretation = image_interpretation.strip()
    else:
        blip2_results = ""

    # Generate clue
    clue_chain = get_clue_chain(
        model=openai_model,
        verbose=verbose)
    clue_results = clue_chain({
        'image_interpretation': image_interpretation,
        'personality': personality,
    })
    ret_dict = {
        'captions': captioning_results,
        'interpretation': image_interpretation,
        'association': clue_results['association'],
        'clue': clue_results['clue'],
    } 
    if blip2_results != "":
        ret_dict['qna_session'] = blip2_results
    else:
        ret_dict['qna_session'] = ""
    if pre_qna_interpretation is not None:
        ret_dict['pre_qna_interpretation'] = pre_qna_interpretation
    else:
        ret_dict['pre_qna_interpretation'] = ""
    return ret_dict


def guess_image_by_clue(images,
                        clue,
                        generate_captions_fn,
                        models,
                        generated_descriptions=None,
                        openai_model='gpt-3.5-turbo-instruct',
                        num_blip2_questions=1,
                        verbose=True):

    results = [dict() for _ in range(len(images))]
    
    for image_idx, image in enumerate(images):
        if generated_descriptions is None:
            # Generate deep captions
            captioning_results = generate_captions_fn(image, models)
            results[image_idx].update({
                'captions': captioning_results["captions"].strip(),
            })

            # Get first interpretation
            image_interp_chain = get_image_interpretation_chain(
                model=openai_model, verbose=verbose)
            image_interpretation = image_interp_chain.predict(
                image_descriptions=captioning_results["captions"],
                ai_models=", ".join(captioning_results["models"]))
            image_interpretation = image_interpretation.strip()
            results[image_idx].update({
                'interpretation': image_interpretation.strip(),
            })

            if num_blip2_questions > 0:
                pre_qna_interpretation = ""  # image_interpretation
                results[image_idx].update({
                    'pre_qna_interpretation': pre_qna_interpretation.strip(),
                })

                # Talk with BLIP-v2 to get more information
                blip2_results = talk_with_blip2(
                    image_interpretation=captioning_results["captions"].strip(),  # image_interpretation,
                    image=image,
                    clue=clue,
                    ask_blip2_fn=models.blip2,
                    num_questions=num_blip2_questions,
                    model=openai_model,
                    verbose=verbose)
                blip2_results = blip2_results.strip()

                # Get final interpretation after QnA session:
                final_interp_chain = get_post_qna_inpterpretation_chain(
                    model=openai_model, verbose=verbose)
                image_interpretation = final_interp_chain.predict(
                    captions=captioning_results["captions"],
                    ai_models=", ".join(captioning_results["models"]),
                    qna_session=blip2_results)
                image_interpretation = image_interpretation.strip()

                results[image_idx].update({
                    'qna_session': blip2_results.strip(),
                    'interpretation': image_interpretation.strip(),
                })
            else:
                results[image_idx].update({
                    'pre_qna_interpretation': image_interpretation.strip(),
                    'qna_session': "",
                })
        else:
            generated_desc = generated_descriptions[image_idx]
            results[image_idx].update({
                'captions': generated_desc['captions']['captions'].strip(),
                'qna_session': generated_desc['qna_session'].strip(),
                'interpretation': generated_desc['interpretation'].strip(),
                'pre_qna_interpretation': generated_desc['pre_qna_interpretation'].strip(),
            })

        # How this image can be related to the cue?
        clue_relation_llm = OpenAI(model_name=openai_model, max_tokens=512)
        clue_relation_prompt = PromptTemplate(
            input_variables=["interpretation", "clue"],
            template=(
                "Given an image with the following description:"
                "\n"
                "{interpretation}"
                "\n"
                'Explain how this image is associated with phrase "{clue}"? '
                "Any movie, book, or historical facts you can think of?"))
        clue_relation_chain = LLMChain(
            llm=clue_relation_llm, prompt=clue_relation_prompt,
            output_key="association", verbose=verbose)
        clue_relation = clue_relation_chain.predict(
            interpretation=results[image_idx]['interpretation'],
            clue=clue)
        results[image_idx].update({
            "clue_relation": clue_relation.strip(),
        })

    # Final step to decide which image suits the clue the best
    prompt = (
        "Given the following image descriptions, explanations how those images "
        'can be associated with the phrase "{clue}", ')

    prompt += "all in the YAML format:\n\n"
    for image_idx in range(len(results)):
        prompt += f"- Image_{image_idx}:\n"
        prompt += f"    description: {results[image_idx]['interpretation']}\n"
        prompt += f"    explanation: {results[image_idx]['clue_relation']}\n"
        
    prompt += (
        "\nWhich image has the best and most logical explanation and "
        'is best described by the phrase "{clue}"? Explain your choice. '
        "Give your final answer as the image name.")

    final_llm = OpenAI(model_name=openai_model, max_tokens=512)
    final_prompt = PromptTemplate(input_variables=["clue"], template=prompt)
    final_prompt_chain = LLMChain(
        llm=final_llm, prompt=final_prompt,
        output_key="answer", verbose=verbose)
    final_answer = final_prompt_chain.predict(clue=clue)
    final_answer = final_answer.strip()
    return {
        'per_image_reasoning': results,
        'final_answer': final_answer,
    }