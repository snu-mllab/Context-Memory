def ppep(entry, lamp_index):
    """Per Profile Entry Prompt"""
    ret = ""
    if lamp_index == 1:
        title = entry["title"]
        ret = f'"{title}"'
    elif lamp_index == 2:
        text = entry["text"]
        category = entry["category"]
        ret = f'the category for the article: "{text}" is "{category}"'
    elif lamp_index == 3:
        score = entry["score"]
        text = entry["text"]
        ret = f'{score} is the score for "{text}"'
    elif lamp_index == 4:
        title = entry["title"]
        text = entry["text"]
        ret = f'"{title}" is the title for "{text}"'
    elif lamp_index == 5:
        title = entry["title"]
        abstract = entry["abstract"]
        ret = f'"{title}" is the title for "{abstract}"'
    elif lamp_index == 7:
        text = entry["text"]
        ret = f'"{text}"'

    return ret

instruction_text = {
    1: "without explanation.",
    2: "article:",
    3: "review:",
    4: "Generate a headline for the following article:",
    5: "Generate a title for the following abstract of a paper:",
    7: "Paraphrase the following tweet without any explanation before or after it:",
}


def ppep_list(profile, lamp_index):
    profile_processed = [ppep(_p, lamp_index=lamp_index) for _p in profile]
    return profile_processed

def split_input(input_, lamp_index):
    instruction_index = input_.rindex(instruction_text[lamp_index]) + len(instruction_text[lamp_index])
    instruction = input_[:instruction_index].strip()
    input_ = input_[instruction_index:].strip()
    return instruction, input_

def generate_instruction(instruction, profile, lamp_index, maybe_comp_str="", sum_tok_str="", gist_profile=False, recur_profile=False):

    conjunction = f"\n{maybe_comp_str}\n, and " if recur_profile else ", and "
    nshot = len(profile)

    if lamp_index == 1:
        n = len("For an author who has written the paper with the title ")
        profile_filtered = [title for title in profile if title not in instruction]
        profile_prompt = conjunction.join(profile_filtered)

        inst_start = ", which reference is related?"
        inst_start_idx = instruction.rindex(inst_start)        
        m = inst_start_idx

        first_part = instruction[:n]
        second_part = instruction[n:m]
        third_part = instruction[m:]
        if len(profile_filtered) > 0:
            instruction = first_part + profile_prompt \
                        + f"\n{maybe_comp_str}{sum_tok_str}\n, and " \
                            + second_part  \
                                + third_part
        
    elif lamp_index == 5:
        profile_prompt = conjunction.join(profile)
        if nshot > 0:
            if recur_profile or gist_profile:
                profile_prompt = f"{profile_prompt}\n{maybe_comp_str}"
                if sum_tok_str:
                    profile_prompt = f"{profile_prompt}{sum_tok_str}"
            instruction = f"{profile_prompt}. Following the given patterns {instruction}"
        # else:
        #     instruction = f"Following the given patterns {instruction}"
    elif lamp_index == 7:
        profile_prompt = conjunction.join(profile)
        if nshot > 0:
            if recur_profile or gist_profile:
                profile_prompt = f"{profile_prompt}\n{maybe_comp_str}"
                if sum_tok_str:
                    profile_prompt = f"{profile_prompt}{sum_tok_str}"
            instruction = f"{profile_prompt} are written by a person. Following the given patterns {instruction}"
        # else:
        #     instruction = f"Following the given patterns {instruction}"
    else:
        profile_prompt = conjunction.join(profile)
        if nshot > 0:
            if recur_profile or gist_profile:
                profile_prompt = f"{profile_prompt}\n{maybe_comp_str}"
                if sum_tok_str:
                    profile_prompt = f"{profile_prompt}{sum_tok_str}"
            instruction = f"{profile_prompt}. {instruction}"

    return instruction


classification_candidates = {
    1: ["[1]", "[2]"],
    2: ["travel", "style & beauty", "food & drink", "sports", "business", "science & technology", "education", "politics", 
    "religion", "crime", "parents", "women", "healthy living", "entertainment", "culture & arts"],
    3: ["1", "2", "3", "4", "5"],
}
