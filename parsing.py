from itertools import takewhile

# stripComments :: [Char] -> String -> String
def stripComments(comment_sequence):
    '''The lines of the input text, with any
       comments (defined as starting with one
       of the characters in comment_sequence) stripped out.
    '''
    def go(cs):
        return lambda s: ''.join(
            takewhile(lambda c: c not in cs, s)
        ).strip()

    return lambda txt: '\n'.join(map(
        go(comment_sequence),
        txt.splitlines()
    ))


def parseMinSec(time_in_min_sec_millis: str) -> float:
    """

    :param time_in_min_sec_millis: [MM:]SS[.Milliseconds] format
    :return: seconds
    """
    minute = seconds = 0

    if ':' in time_in_min_sec_millis:
        left, right = time_in_min_sec_millis.split(":")
        minute = float(left) * 60
        seconds = float(right)
    else:
        seconds = float(time_in_min_sec_millis)

    return minute + seconds