import re


STATES = {
    'AK':'Alaska',
    'AL':'Alabama',
    'AR':'Arkansas',
    'AZ':'Arizona',
    'CA':'California',
    'CO':'Colorado',
    'CT':'Connecticut',
    'DC':'District of Columbia',
    'DE':'Delaware',
    'FL':'Florida',
    'GA':'Georgia',
    'HI':'Hawaii',
    'IA':'Iowa',
    'ID':'Idaho',
    'IL':'Illinois',
    'IN':'Indiana',
    'KS':'Kansas',
    'KY':'Kentucky',
    'LA':'Louisiana',
    'MA':'Massachusetts',
    'MD':'Maryland',
    'ME':'Maine',
    'MI':'Michigan',
    'MN':'Minnesota',
    'MO':'Missouri',
    'MS':'Mississippi',
    'MT':'Montana',
    'NC':'North Carolina',
    'ND':'North Dakota',
    'NE':'Nebraska',
    'NH':'New Hampshire',
    'NJ':'New Jersey',
    'NM':'New Mexico',
    'NV':'Nevada',
    'NY':'New York',
    'OH':'Ohio',
    'OK':'Oklahoma',
    'OR':'Oregon',
    'PA':'Pennsylvania',
    'RI':'Rhode Island',
    'SC':'South Carolina',
    'SD':'South Dakota',
    'TN':'Tennessee',
    'TX':'Texas',
    'UT':'Utah',
    'VA':'Virginia',
    'VT':'Vermont',
    'WA':'Washington',
    'WI':'Wisconsin',
    'WV':'West Virginia',
    'WY':'Wyoming'
}


def get_keyword_matches(result, correct_keywords, return_texts=False):
    match_texts = []
    matches = 0
    if not result:
        if return_texts:
            return matches, match_texts
        return matches
    for keyword in correct_keywords:
        keyword_re = rf"[(\b\s]{keyword}[,\s\b]"
        # dollar amounts look for the full int sans symbols
        if isinstance(keyword, (int, float)) or str(keyword).startswith("$"):
            res_nocomma = re.sub(r"[$,]+", "", result)
            found = re.findall(keyword_re, res_nocomma, re.I)
            if len(found) > 0:
                matches += 1
                match_texts.append(found)
        # # if we have a state, check for case-sensitive abbrev + full name
        # elif keyword in STATES:
        #     if f" {keyword}" in result:
        #         matches += 1
        #     elif STATES[keyword] in result:
        #         matches += 1
        # case insensitive match on phrases
        else:
            found = re.findall(keyword_re, result, re.I)
            if len(found) > 0:
                matches += 1
                match_texts.append(found)
    if return_texts:
        return matches, match_texts
    return matches
