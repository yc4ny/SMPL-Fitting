def solution(s):
    splitted = s.split(" ")
    answer_list = []
    if " " in splitted:
        splitted.remove(" ")
    
    for i in range(len(splitted)):
        if len(splitted[i]) > 0:
            if splitted[i][0].isdigit():
                answer_list.append([splitted[i][:].lower()])

            else:
                answer_list.append([splitted[i][0].upper(),splitted[i][1:].lower()])
                
    answer = ""
    for i in range(len(answer_list)):
        for j in range(len(answer_list[i])):
            answer += answer_list[i][j]
        if i != len(answer_list)-1:
            answer += " "
    
    return answer

if __name__ == "__main__":
    s = "3people unFollowed me"	
    s = "for   the   last week"

    print(solution(s))