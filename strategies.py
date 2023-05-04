"""
Some example strategies for people who want to create a custom, homemade bot.

With these classes, bot makers will not have to implement the UCI or XBoard interfaces themselves.
"""

from __future__ import annotations
import chess
from chess.engine import PlayResult
import random
from engine_wrapper import MinimalEngine
from typing import Any


class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""

    pass


# Strategy names and ideas from tom7's excellent eloWorld video

class RandomMove(ExampleEngine):
    """Get a random move."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose a random move."""
        return PlayResult(random.choice(list(board.legal_moves)), None)


class Alphabetical(ExampleEngine):
    """Get the first move when sorted by san representation."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose the first move alphabetically."""
        moves = list(board.legal_moves)
        moves.sort(key=board.san)
        return PlayResult(moves[0], None)


class FirstMove(ExampleEngine):
    """Get the first move when sorted by uci representation."""

    def search(self, board: chess.Board, *args: Any) -> PlayResult:
        """Choose the first move alphabetically in uci representation."""
        moves = list(board.legal_moves)
        moves.sort(key=str)
        return PlayResult(moves[0], None)

class sgai(MinimalEngine):

    def load(self, net:nn.NeuralNet):
        net.loadbiases("bias.p")
        net.loadweights("weights.p")   

    def boardtobitstring(self, board):
        p = ""
        p = p + '{0:064b}'.format(int(board.pieces(chess.KING, chess.WHITE)))[::-1]
        p = p + '{0:064b}'.format(int(board.pieces(chess.QUEEN, chess.WHITE)))[::-1]
        p = p + '{0:064b}'.format(int(board.pieces(chess.ROOK, chess.WHITE)))[::-1]
        p = p + '{0:064b}'.format(int(board.pieces(chess.BISHOP, chess.WHITE)))[::-1]
        p = p + '{0:064b}'.format(int(board.pieces(chess.KNIGHT, chess.WHITE)))[::-1]
        p = p + '{0:064b}'.format(int(board.pieces(chess.PAWN, chess.WHITE)))[::-1]
        p = p + '{0:064b}'.format(int(board.pieces(chess.KING, chess.BLACK)))[::-1]
        p = p + '{0:064b}'.format(int(board.pieces(chess.QUEEN, chess.BLACK)))[::-1]
        p = p + '{0:064b}'.format(int(board.pieces(chess.ROOK, chess.BLACK)))[::-1]
        p = p + '{0:064b}'.format(int(board.pieces(chess.BISHOP, chess.BLACK)))[::-1]
        p = p + '{0:064b}'.format(int(board.pieces(chess.KNIGHT, chess.BLACK)))[::-1]
        p = p + '{0:064b}'.format(int(board.pieces(chess.PAWN, chess.BLACK)))[::-1]
        p = p + '{0:040b}'.format(int(0))

        return bitstring.BitArray(bin=p)


    def search(self, board, *args):
        net = nn.NeuralNet()
        self.load(net)
        moves = list(board.legal_moves)
        position = chess.Bitboard()
        topeval = -1000

        for i in moves:
            boardcopy = board.copy()
            boardcopy.push(i)

            net.loadinputbits(self.boardtobitstring(boardcopy))
            evaluation = net.calcoutput()
            if(evaluation > topeval):
                topeval = evaluation
                topmove = i
        return PlayResult(topmove)
