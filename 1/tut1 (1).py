from sys import argv, exit


CIRCUIT = ".circuit"
END = ".end"


def line2tokens(spiceLine):
    tw = spiceLine.split()

    # R, L, C, Independent Sources
    if len(tw) == 4:
        elementName = tw[0]
        node1 = tw[1]
        node2 = tw[2]
        value = tw[3]
        return [elementName, node1, node2, value]

    # CCxS
    elif len(tw) == 5:
        elementName = tw[0]
        node1 = tw[1]
        node2 = tw[2]
        voltageSource = tw[3]
        value = tw[4]
        return [elementName, node1, node2, voltageSource, value]

    # VCxS
    elif len(tw) == 6:
        elementName = tw[0]
        node1 = tw[1]
        node2 = tw[2]
        voltageSourceNode1 = tw[3]
        voltageSourceNode2 = tw[4]
        value = tw[5]
        return [
            elementName,
            node1,
            node2,
            voltageSourceNode1,
            voltageSourceNode2,
            value,
        ]

    else:
        return []


def getToken(lines):
    lines_token = []
    for i in range(0, len(lines)):  # iterate over valid range
        line = (
            lines[i].split("#")[0].split()
        )  # remove comment and split line into words

        lines_token.append(
            line
        )  # join words after reversing and add "\n" at the end of line

    return lines_token

def pw(lines):
    output = ""
    for i in reversed(range(0, len(lines))):  # iterate over valid range
        line = (
            lines[i].split("#")[0].split()
        )  # remove comment and split line into words

        line.reverse()  # reverse the list
        output = output + (
            " ".join(line) + "\n"
        )  # join words after reversing and add "\n" at the end of line

    print(output)


if len(argv) != 2:
    print("Invalid operation !")
    print(f"Usage: {argv[0]} <inputfile>'")
    exit()

try:
    with open(argv[1]) as f:
        lines = f.readlines()
        start = -1
        end = -2
        cnt_s = 0
        cnt_a = 0
        for line in lines:  # extracting circuit definition: start and end lines
            if CIRCUIT == line[0 : len(CIRCUIT)]:
                start = lines.index(line)
                cnt_s = cnt_s + 1
            elif END == line[: len(END)]:
                end = lines.index(line)
                cnt_a = cnt_a + 1

        if (cnt_s > 1 or cnt_a > 1):  # Check if there are multiple .circuit/.end declarations in input file
            print(
                "Invalid circuit definition! Multiple .circuit/.end declarations detected"
            )
            exit(0)

        Lines = lines[start + 1 : end]
        LinesToken = getToken(Lines)
        pw(Lines)


except IOError:
    print("Invalid file")
    exit()
    



