import re


def extract_layer_idx_from_state_dict(state_dict):
    extracted_numbers = []
    for key in state_dict["feature_map_model_state_dict"].keys():
        matches = re.findall(r"\.(\d+)\.", key)
        if matches:
            extracted_numbers.extend(int(match) for match in matches)
    return extracted_numbers
