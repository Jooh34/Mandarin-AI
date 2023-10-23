class Move:
    @staticmethod
    def get_possible_actions(_board, turn):
        actions = [] # (x, y)
        m = len(_board)
        n = len(_board[0])
        for i in range(m):
            for j in range(n):
                if Move.is_action_possible(_board, turn, i, j):
                    actions.append((i,j))
        
        if not actions:
            actions.append((-1,-1)) # pass move

        return actions

    @staticmethod
    def is_action_possible(_board, turn, x, y):
        if _board[x][y] != 0: # piece exists
            return False
        
        m = len(_board)
        n = len(_board[0])
        dirs = [(-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dx,dy in dirs:
            enemy_piece_exist = False
            nx = x+dx; ny = y+dy

            while 0 <= nx < m and 0 <= ny < n:
                if _board[nx][ny] == -turn: # enemy piece
                    enemy_piece_exist = True

                elif _board[nx][ny] == turn: # my piece
                    if enemy_piece_exist:
                        return True # find action!
                    else:
                        break
                else: # empty
                    break

                nx += dx
                ny += dy

        # checked all directions
        return False 

    @staticmethod
    def take_action(_board, turn, action):
        x,y = action
        if x == -1 and y == -1: # pass move
            return 0

        if _board[x][y] != 0: # piece exists
            raise Exception(f"error take_action : piece already exists on board [{x}][{y}]")
        
        m = len(_board)
        n = len(_board[0])
        dirs = [(-1, 0), (-1, -1), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        flip_count = 0
        for dx,dy in dirs:
            enemy_pieces = []
            nx = x+dx; ny = y+dy

            while 0 <= nx < m and 0 <= ny < n:
                if _board[nx][ny] == -turn: # enemy piece
                    enemy_pieces.append((nx,ny))

                elif _board[nx][ny] == turn: # my piece
                    if enemy_pieces:
                        flip_count += len(enemy_pieces)
                        for ex, ey in enemy_pieces: # flip!
                            _board[ex][ey] *= (-1)
                        break

                    else:
                        break
                else: # empty
                    break

                nx += dx
                ny += dy

        _board[x][y] = turn
        return flip_count