import re

from core.board import Board
from core.types import Piece, MAX_ROW, MAX_COL, Camp, Formation
from copy import deepcopy

class Gibo:
    '''
        match_info : name, han_formation, cho_formation, winner, move_counts ...
        moves : [move(str) ...]
        board_history : [board(Board) ... ]
    '''
    KEY_CHO_PLAYER = "초대국자"
    KEY_HAN_PLAYER = "한대국자"
    KEY_CHO_FORMATION = "초차림"
    KEY_HAN_FORMATION = "한차림"
    KEY_TOTAL_MOVE = "총수"
    KEY_MATCH_RESULT = "대국결과"

    def __init__(self, match_info, move_history, board_history):
        self.match_info = match_info
        self.move_history = move_history
        self.board_history = board_history

    @staticmethod
    def make_gibo_with_gib(file_path):
        match_info = {}
        move_history = []

        with open(file_path, encoding='euc-kr') as f:
            for line in f:
                m = re.match("\[(.*) \"(.*)\"\]", line)
                if m: # meta
                    try:
                        match_info[m.group(1)] = m.group(2)
                    except:
                        raise Exception(f'match_info parse error : wrong line in gib file : {line} \n in {file_path}')


                else:
                    move_match = re.findall("(\d+. [0-9]{2}.[0-9]{2})", line)
                    if move_match: #moves
                        for move in move_match:
                            index, s = move.split('. ')
                            if int(index) != len(move_history)+1: # gib file error
                                raise Exception(f'index != movehistory not matched. wrong line in gib file : {line} \n in {file_path}')

                            move_history.append(s)

        board_history = Gibo.make_board_history(match_info, move_history)
        return Gibo(match_info, move_history, board_history)
    
    @staticmethod
    def make_board_history(match_info, move_history):
        kr_to_piece_table = {
            '졸' : Piece.CHO_ZOL,
            '병' : Piece.CHO_ZOL,
            '상' : Piece.CHO_SANG,
            '사' : Piece.CHO_SA,
            '마' : Piece.CHO_MA,
            '포' : Piece.CHO_PO,
            '차' : Piece.CHO_CHA,
            '장' : Piece.CHO_GOONG,
        }

        cho_form = Formation.str_to_formation(match_info[Gibo.KEY_CHO_FORMATION]) 
        han_form = Formation.str_to_formation(match_info[Gibo.KEY_HAN_FORMATION]) 
        board = Board(cho_form, han_form)

        board_history = []
        board_history.append(deepcopy(board))

        total_move = int(match_info[Gibo.KEY_TOTAL_MOVE])

        for i, move in enumerate(move_history):
            camp = Camp.CHO if i % 2 == 0 else Camp.HAN

            r1,c1,p,r2,c2= list(move)
            r1,c1,r2,c2 = map(int, [r1,c1,r2,c2])
            p = move[2]

            # adjust index
            r1 = (r1-1+MAX_ROW) % MAX_ROW
            r2 = (r2-1+MAX_ROW) % MAX_ROW
            c1 -= 1
            c2 -= 1

            piece = kr_to_piece_table[p]
            piece = piece * camp

            board.set(r1,c1,0)
            board.set(r2,c2,piece)
            board_history.append(deepcopy(board))

        if len(board_history) != total_move+1:
            raise Exception(f'board history : {len(board_history)} != total move+1 : {total_move+1}.')

        return board_history



# example gibo
'''
[대회명 "친선대국"]
[회전 "미상"]
[대국일자 "2016-12-06"]
[대국장소 "미상"]
[초대국자 "하여명"]
[한대국자 "전동하"]
[초차림 "상마상마"]
[한차림 "상마상마"]
[제한시간 "미상"]
[총수 "73"]
[대국결과 "초 완승"]

1. 71졸72 2. 49병48 3. 08마87 4. 18마37 5. 88포85 6. 38포35
7. 75졸76 8. 13마34 9. 07상75 10. 43병42 11. 02상74 12. 12상44
13. 74상42병 14. 44상72졸 15. 03마84 16. 72상44 17. 42상65 18. 35포55
19. 84마63 20. 32포35 21. 09차08 22. 19차18 23. 73졸74 24. 11차13
25. 74졸64 26. 13차53 27. 01차41병 28. 47병57 29. 95장05 30. 14사24
31. 06사95 32. 35포15 33. 85포89 34. 17상49 35. 89포83 36. 53차52
37. 83포88 38. 18차17 39. 63마44상 40. 45병44마 41. 75상52차 42. 25장14
43. 41차44병 44. 49상77졸 45. 52상24사 46. 14장24상 47. 88포84 48. 16사25
49. 08차48병 50. 37마56 51. 44차47 52. 17차16 53. 84포06 54. 16차19
55. 76졸77상 56. 56마77졸 57. 47차57병 58. 77마85 59. 95사85마 60. 55포85사
61. 05장95 62. 85포55장군 63. 95장94 64. 19차16 65. 82포89 66. 16차76
67. 89포84 68. 76차74 69. 64졸63 70. 25사35 71. 57차27장군 72. 35사25
73. 48차41
'''

if __name__ == '__main__':
    Gibo.make_gibo_with_gib("E:\work\Mandarin-AI\data\햇터몰배 제2회 세계인 장기대회 2(하여명 vs 전동하).gib")