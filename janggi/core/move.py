from core.types import Piece, Util, MoveType

class Action:
    def __init__(self, prev, next, piece, move_type):
        self.prev = prev
        self.next = next
        self.piece = piece
        self.move_type = move_type
    
    def __str__(self):
        print(self.piece)
        return f'{self.prev[0]}{self.prev[1]}{Util.piece_to_kor(self.piece)}{self.next[0]}{self.next[1]} {self.move_type}'
    
    def is_prev(self,x,y):
        return x==self.prev[0] and y==self.prev[1]

    def is_prev(self,tu):
        return tu[0]==self.prev[0] and tu[1]==self.prev[1]

    def is_next(self,x,y):
        return x==self.next[0] and y==self.next[1]

    def is_next(self,tu):
        return tu[0]==self.next[0] and tu[1]==self.next[1]
        
class Move:
    sang_moves = {
        MoveType.SANG_UPLEFT : [[-1,0], [-1,-1], [-1,-1]], # up-left
        MoveType.SANG_UPLEFT+1 : [[-1,0], [-1,1], [-1,1]], #up-right
        MoveType.SANG_UPLEFT+2: [[0,1], [-1,1], [-1,1]], #right-up
        MoveType.SANG_UPLEFT+3: [[0,1], [1,1], [1,1]], # right-down
        MoveType.SANG_UPLEFT+4: [[1,0], [1,1], [1,1]], # down-right
        MoveType.SANG_UPLEFT+5: [[1,0], [1,-1], [1,-1]], #down-left
        MoveType.SANG_UPLEFT+6: [[0,-1], [1,-1], [1,-1]], #left-down
        MoveType.SANG_UPLEFT+7: [[0,-1], [-1,-1], [-1,-1]], #left-up
    }

    ma_moves = {
        MoveType.MA_UPLEFT: [[-1,0], [-1,-1]], # up-left
        MoveType.MA_UPLEFT+1: [[-1,0], [-1,1]], #up-right
        MoveType.MA_UPLEFT+2: [[0,1], [-1,1]], #right-up
        MoveType.MA_UPLEFT+3: [[0,1], [1,1]], # right-down
        MoveType.MA_UPLEFT+4: [[1,0], [1,1]], # down-right
        MoveType.MA_UPLEFT+5: [[1,0], [1,-1]], #down-left
        MoveType.MA_UPLEFT+6: [[0,-1], [1,-1]], #left-down
        MoveType.MA_UPLEFT+7: [[0,-1], [-1,-1]], #left-up
    }

    palace_move_table = {
        MoveType.MOVE_UP: (-1,0),
        MoveType.MOVE_DOWN: (1,0),
        MoveType.MOVE_LEFT: (0,-1),
        MoveType.MOVE_RIGHT: (0,1),
        MoveType.DIAG_UPLEFT: (-1,-1),
        MoveType.DIAG_UPRIGHT: (-1,1),
        MoveType.DIAG_DOWNRIGHT: (1,1),
        MoveType.DIAG_DOWNLEFT: (1,-1),
    }

    @staticmethod
    def get_palace_move_types(i,j):
        dirs = []
        if 7 <= i <= 9 and 3 <= j <= 5: # cho palace
            if 8 <= i <= 9:
                dirs.append(MoveType.MOVE_UP)
            
            if 7 <= i <= 8:
                dirs.append(MoveType.MOVE_DOWN)

            if 3 <= j <= 4:
                dirs.append(MoveType.MOVE_RIGHT)

            if 4 <= j <= 5:
                dirs.append(MoveType.MOVE_LEFT)

            if i==8 and j==4:
                dirs.extend([MoveType.DIAG_UPLEFT, MoveType.DIAG_UPRIGHT, MoveType.DIAG_DOWNRIGHT, MoveType.DIAG_DOWNLEFT])

            if i==7 and j==3:
                dirs.append(MoveType.DIAG_DOWNRIGHT)

            if i==7 and j==5:
                dirs.append(MoveType.DIAG_DOWNLEFT)

            if i==9 and j==3:
                dirs.append(MoveType.DIAG_UPRIGHT)

            if i==9 and j==5:
                dirs.append(MoveType.DIAG_UPLEFT)

        elif 0 <= i <= 2 and 3 <= j <= 5: # han palace
            if 1 <= i <= 2:
                dirs.append(MoveType.MOVE_UP)
            
            if 0 <= i <= 1:
                dirs.append(MoveType.MOVE_DOWN)

            if 3 <= j <= 4:
                dirs.append(MoveType.MOVE_RIGHT)

            if 4 <= j <= 5:
                dirs.append(MoveType.MOVE_LEFT)

            if i==1 and j==4:
                dirs.extend([MoveType.DIAG_UPLEFT, MoveType.DIAG_UPRIGHT, MoveType.DIAG_DOWNRIGHT, MoveType.DIAG_DOWNLEFT])

            if i==0 and j==3:
                dirs.append(MoveType.DIAG_DOWNRIGHT)

            if i==0 and j==5:
                dirs.append(MoveType.DIAG_DOWNLEFT)

            if i==2 and j==3:
                dirs.append(MoveType.DIAG_UPRIGHT)

            if i==2 and j==5:
                dirs.append(MoveType.DIAG_UPLEFT)
        else:
            raise Exception('get_palace_movement invalid input.')
        
        return dirs
    
    @staticmethod
    def is_in_cho_palace(i,j):
        if 7 <= i <= 9 and 3 <= j <= 5:
            return True
        return False

    @staticmethod
    def is_in_han_palace(i,j):
        if 0 <= i <= 2 and 3 <= j <= 5:
            return True
        return False
    
    @staticmethod
    def is_in_palace(i,j):
        return Move.is_in_cho_palace(i,j) or Move.is_in_han_palace(i,j)

    @staticmethod
    def get_possible_actions(_board):
        actions = [] # (from_x, from,y, to_x, to_y, piece)
        m = len(_board)
        n = len(_board[0])
        for i in range(m):
            for j in range(n):
                actions.extend(Move.get_possible_piece_actions(_board,i,j))
        
        return actions

    @staticmethod
    def get_possible_piece_actions(_board,i,j):
        if _board[i][j] <= 0: # not my piece
            return []

        m = len(_board)
        n = len(_board[0])
        actions = []
        if _board[i][j] == Piece.CHO_ZOL:
            dirs = [(-1,0,MoveType.MOVE_UP),(0,-1,MoveType.MOVE_LEFT),(0,1,MoveType.MOVE_RIGHT)] # dx, dy, move_type
            for dx,dy,move_type in dirs:
                if 0 <= i+dx < m and 0 <= j+dy < n:
                    if not Piece.is_mine(_board[i+dx][j+dy]):
                        actions.append(Action([i,j], [i+dx,j+dy], _board[i][j], move_type))
            
            # actions in palace
            if Move.is_in_palace(i,j):
                move_types = Move.get_palace_move_types(i,j)
                to_removes = [MoveType.MOVE_DOWN, MoveType.DIAG_DOWNLEFT, MoveType.DIAG_DOWNRIGHT]
                for to_remove in to_removes:
                    try: move_types.remove(to_remove)
                    except ValueError: pass
                
                for move_type in move_types:
                    dx,dy = Move.palace_move_table[move_type]
                    if not Piece.is_mine(_board[i+dx][j+dy]):
                        actions.append(Action([i,j], [i+dx,j+dy], _board[i][j], move_type))
        
        elif _board[i][j] == Piece.CHO_SANG:
            for move_type, move in Move.sang_moves.items():
                next_i = i; next_j = j
                for k,(dx,dy) in enumerate(move):
                    next_i += dx; next_j += dy
                    if 0 <= next_i < m and 0 <= next_j < n:
                        if k == 2 and not Piece.is_mine(_board[next_i][next_j]): # destination and not my piece
                            actions.append(Action([i,j], [next_i,next_j], _board[i][j], move_type))

                        elif k < 2 and not Piece.is_empty(_board[next_i][next_j]):
                            break # blocked Myuck

                    else:
                        break # out of _board

        elif _board[i][j] == Piece.CHO_MA:
            for move_type, move in Move.ma_moves.items():
                next_i = i; next_j = j
                for k,(dx,dy) in enumerate(move):
                    next_i += dx; next_j += dy
                    if 0 <= next_i < m and 0 <= next_j < n:
                        if k == 1 and not Piece.is_mine(_board[next_i][next_j]): # destination and not my piece
                            actions.append(Action([i,j], [next_i,next_j], _board[i][j], move_type))

                        elif k < 1 and not Piece.is_empty(_board[next_i][next_j]):
                            break # blocked Myuck

                    else:
                        break # out of _board
        
        elif _board[i][j] == Piece.CHO_PO:
            dirs = [(-1,0,MoveType.MOVE_UP), (1,0,MoveType.MOVE_DOWN), (0,-1,MoveType.MOVE_LEFT),(0,1,MoveType.MOVE_RIGHT)] # dx, dy, move_type
            for dx,dy,move_type in dirs:
                next_i = i+dx; next_j = j+dy
                cnt = 0; jumped = False
                while True:
                    if 0 <= next_i < m and 0 <= next_j < n:
                        if jumped:
                            if Piece.is_enemy(_board[next_i][next_j]):
                                if not Piece.is_po(_board[next_i][next_j]): # po can't take po
                                    actions.append(Action([i,j], [next_i,next_j], _board[i][j], move_type+cnt))
                                break
                            elif Piece.is_empty(_board[next_i][next_j]):
                                actions.append(Action([i,j], [next_i,next_j], _board[i][j], move_type+cnt))
                            else: # Piece is mine
                                break
                        
                        else: # not jumped
                            if Piece.is_empty(_board[next_i][next_j]):
                                pass
                            elif Piece.is_po(_board[next_i][next_j]):
                                break # blocked by po
                            else:
                                jumped = True

                    else: # out of _board
                        break 
                    
                    next_i+=dx; next_j+=dy; cnt+=1
            
            # actions in palace
            if Move.is_in_palace(i,j):
                move_types = Move.get_palace_move_types(i,j)
                for move_type in move_types:
                    dx,dy = Move.palace_move_table[move_type]
                    next_i = i+dx; next_j = j+dy
                    cnt = 0; jumped = False
                    while True:
                        if Move.is_in_palace(next_i, next_j):
                            if jumped:
                                if Piece.is_enemy(_board[next_i][next_j]):
                                    if not Piece.is_po(_board[next_i][next_j]): # po can't take po
                                        actions.append(Action([i,j], [next_i,next_j], _board[i][j], move_type+cnt))
                                    break
                                elif Piece.is_empty(_board[next_i][next_j]):
                                    actions.append(Action([i,j], [next_i,next_j], _board[i][j], move_type+cnt))
                                else: # Piece is mine
                                    break
                        
                            else: # not jumped
                                if Piece.is_empty(_board[next_i][next_j]):
                                    pass
                                elif Piece.is_po(_board[next_i][next_j]):
                                    break # blocked by po
                                else:
                                    jumped = True

                        else: # Piece out of palace
                            break
                        
                        next_i+=dx; next_j+=dy; cnt+=1

        elif _board[i][j] == Piece.CHO_CHA:
            dirs = [(-1,0,MoveType.MOVE_UP), (1,0,MoveType.MOVE_DOWN), (0,-1,MoveType.MOVE_LEFT),(0,1,MoveType.MOVE_RIGHT)] # dx, dy, move_type
            for dx,dy,move_type in dirs:
                next_i = i+dx; next_j = j+dy
                cnt = 0
                while True:
                    if 0 <= next_i < m and 0 <= next_j < n:
                        if Piece.is_enemy(_board[next_i][next_j]):
                            actions.append(Action([i,j], [next_i,next_j], _board[i][j], move_type+cnt))
                            break
                        elif Piece.is_empty(_board[next_i][next_j]):
                            actions.append(Action([i,j], [next_i,next_j], _board[i][j], move_type+cnt))
                        else: # Piece is mine
                            break
                    else:
                        break # out of _board

                    next_i+=dx; next_j+=dy; cnt+=1
            
            # actions in palace
            if Move.is_in_palace(i,j):
                move_types = Move.get_palace_move_types(i,j)
                for move_type in move_types:
                    dx,dy = Move.palace_move_table[move_type]
                    next_i = i+dx; next_j = j+dy
                    cnt = 0
                    while True:
                        if Move.is_in_palace(next_i, next_j):
                            if Piece.is_enemy(_board[next_i][next_j]):
                                actions.append(Action([i,j], [next_i,next_j], _board[i][j], move_type+cnt))
                                break
                            elif Piece.is_empty(_board[next_i][next_j]):
                                actions.append(Action([i,j], [next_i,next_j], _board[i][j], move_type+cnt))
                            else: # Piece is mine
                                break
                        else: # Piece out of palace
                            break

                        next_i+=dx; next_j+=dy; cnt+=1
        
        elif _board[i][j] == Piece.CHO_GOONG or _board[i][j] == Piece.CHO_SA:
            move_types = Move.get_palace_move_types(i,j)
            for move_type in move_types:
                dx,dy = Move.palace_move_table[move_type]
                if not Piece.is_mine(_board[i+dx][j+dy]):
                    actions.append(Action([i,j], [i+dx,j+dy], _board[i][j], move_type))

        return actions
        