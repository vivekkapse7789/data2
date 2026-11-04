N = 8


def is_safe(board, row, col):
    """Check that no queen already placed in rows above attacks (row, col)."""
    # Same column
    for r in range(row):
        if board[r][col] == 1:
            return False
    # Upper-left diagonal
    r, c = row - 1, col - 1
    while r >= 0 and c >= 0:
        if board[r][c] == 1:
            return False
        r, c = r - 1, c - 1
    # Upper-right diagonal
    r, c = row - 1, col + 1
    while r >= 0 and c < N:
        if board[r][c] == 1:
            return False
        r, c = r - 1, c + 1
    return True


def solve(board, row):
    """Place queens row by row using backtracking."""
    if row == N:                      # all queens placed
        return True
    for col in range(N):
        if is_safe(board, row, col):
            board[row][col] = 1       # place queen
            if solve(board, row + 1):
                return True
            board[row][col] = 0       # backtrack
    return False


def print_board(board):
    for row in board:
        print(" ".join("Q" if cell else "." for cell in row))


def print_matrix(board):
    for row in board:
        print(row)


def eight_queens(first_col):
    board = [[0] * N for _ in range(N)]
    board[0][first_col] = 1           # first queen is fixed in row 0
    if solve(board, 1):               # backtrack for remaining 7 queens
        return board
    return None


if __name__ == "__main__":
    first_col = int(input(f"Enter column (0-{N - 1}) for the first queen in row 0: "))
    if not 0 <= first_col < N:
        raise SystemExit("Invalid column.")

    result = eight_queens(first_col)
    if result:
        print("\nFinal 8-Queens matrix:")
        print_matrix(result)
        print("\nBoard view:")
        print_board(result)
    else:
        print("No solution exists with the first queen at that position.")
