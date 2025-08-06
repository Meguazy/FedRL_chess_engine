from dataclasses import dataclass
from typing import List, Optional

import chess
import numpy as np


@dataclass
class OpeningTemplate:
    """Template for specific opening lines."""
    name: str
    eco_code: str
    moves: List[str]  # Move sequence in algebraic notation
    style_category: str  # 'tactical', 'positional', 'dynamic'
    frequency_weight: float  # How often to play this opening
    continuation_depth: int  # How many moves to force


class ECO_OpeningDatabase:
    """Database of ECO openings organized by style characteristics."""
    
    def __init__(self):
        self.openings_by_style = {
            'tactical': self._create_tactical_openings(),
            'positional': self._create_positional_openings(), 
            'dynamic': self._create_dynamic_openings()
        }
        
        # Validate opening moves
        self._validate_all_openings()
    
    def _create_tactical_openings(self) -> List[OpeningTemplate]:
        """Create tactical opening templates (sharp, aggressive)."""
        return [
            # Sicilian Defence variations (Volume B)
            OpeningTemplate("Sicilian Dragon", "B70", 
                           ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6"], 
                           "tactical", 0.15, 10),
            
            OpeningTemplate("Sicilian Najdorf", "B90",
                           ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"],
                           "tactical", 0.12, 10),
            
            OpeningTemplate("Sicilian Accelerated Dragon", "B35",
                           ["e4", "c5", "Nf3", "g6", "d4", "cxd4", "Nxd4", "Bg7", "Nc3", "Nc6"],
                           "tactical", 0.10, 10),
            
            # King's Gambit (Volume C)
            OpeningTemplate("King's Gambit Accepted", "C33",
                           ["e4", "e5", "f4", "exf4", "Nf3", "g5", "h4", "g4", "Ne5"],
                           "tactical", 0.08, 9),
            
            OpeningTemplate("King's Gambit Declined", "C30",
                           ["e4", "e5", "f4", "Bc5", "Nf3", "d6", "c3", "f5"],
                           "tactical", 0.06, 8),
            
            # Italian Game aggressive lines (Volume C)
            OpeningTemplate("Evans Gambit", "C51",
                           ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "b4", "Bxb4", "c3", "Ba5", "d4"],
                           "tactical", 0.07, 11),
            
            OpeningTemplate("Italian Game Aggressive", "C50",
                           ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "d3", "f5", "exf5", "Bxf2+"],
                           "tactical", 0.08, 10),
            
            # Alekhine's Defence (Volume B)  
            OpeningTemplate("Alekhine Defence Four Pawns", "B03",
                           ["e4", "Nf6", "e5", "Nd5", "d4", "d6", "c4", "Nb6", "f4", "dxe5", "fxe5"],
                           "tactical", 0.06, 11),
            
            # Vienna Game (Volume C)
            OpeningTemplate("Vienna Gambit", "C25",
                           ["e4", "e5", "Nc3", "Nc6", "f4", "exf4", "Nf3", "g5", "h4", "g4", "Ng5"],
                           "tactical", 0.05, 11),
            
            # Centre Game (Volume C)
            OpeningTemplate("Centre Game", "C21", 
                           ["e4", "e5", "d4", "exd4", "Qxd4", "Nc6", "Qa4", "d6", "Nf3", "Bd7"],
                           "tactical", 0.04, 10),
            
            # Scandinavian Defence (Volume B)
            OpeningTemplate("Scandinavian Main Line", "B01",
                           ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qa5", "d4", "Nf6", "Nf3", "Bf5"],
                           "tactical", 0.05, 10)
        ]
    
    def _create_positional_openings(self) -> List[OpeningTemplate]:
        """Create positional opening templates (closed, strategic)."""
        return [
            # Queen's Gambit variations (Volume D)
            OpeningTemplate("Queen's Gambit Declined Orthodox", "D63",
                           ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O", "Nf3"],
                           "positional", 0.15, 11),
            
            OpeningTemplate("Queen's Gambit Declined Tarrasch", "D34",
                           ["d4", "d5", "c4", "e6", "Nc3", "c5", "cxd5", "exd5", "Nf3", "Nc6", "g3"],
                           "positional", 0.10, 11),
            
            OpeningTemplate("Slav Defence", "D10",
                           ["d4", "d5", "c4", "c6", "Nc3", "Nf6", "e3", "Bf5", "Nf3", "e6", "Nh4"],
                           "positional", 0.12, 11),
            
            # Indian Defences (Volume E)
            OpeningTemplate("Nimzo-Indian Defence", "E20",
                           ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3", "c5", "Bd3", "Nc6", "Nf3"],
                           "positional", 0.14, 11),
            
            OpeningTemplate("Queen's Indian Defence", "E12",
                           ["d4", "Nf6", "c4", "e6", "Nf3", "b6", "g3", "Ba6", "b3", "Bb4+", "Bd2"],
                           "positional", 0.11, 11),
            
            OpeningTemplate("Catalan Opening", "E00",
                           ["d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "Be7", "Nf3", "O-O", "O-O"],
                           "positional", 0.13, 11),
            
            # English Opening positional lines (Volume A)
            OpeningTemplate("English Opening Symmetrical", "A30",
                           ["c4", "c5", "Nf3", "Nf6", "g3", "b6", "Bg2", "Bb7", "O-O", "g6", "Nc3"],
                           "positional", 0.09, 11),
            
            OpeningTemplate("English Opening Closed System", "A25",
                           ["c4", "e5", "Nc3", "Nc6", "g3", "g6", "Bg2", "Bg7", "d3", "d6", "Nf3"],
                           "positional", 0.08, 11),
            
            # Réti Opening (Volume A)
            OpeningTemplate("Réti Opening", "A09",
                           ["Nf3", "d5", "c4", "c6", "b3", "Bf5", "Bb2", "Nf6", "g3", "e6", "Bg2"],
                           "positional", 0.07, 11),
            
            # Bogo-Indian Defence (Volume E)
            OpeningTemplate("Bogo-Indian Defence", "E11",
                           ["d4", "Nf6", "c4", "e6", "Nf3", "Bb4+", "Bd2", "Qe7", "g3", "Nc6", "Bg2"],
                           "positional", 0.06, 11)
        ]
    
    def _create_dynamic_openings(self) -> List[OpeningTemplate]:
        """Create dynamic opening templates (flexible, imbalanced)."""
        return [
            # King's Indian Defence (Volume E)
            OpeningTemplate("King's Indian Classical", "E90",
                           ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O", "Be2"],
                           "dynamic", 0.16, 11),
            
            OpeningTemplate("King's Indian Sämisch", "E80",
                           ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "f3", "O-O", "Be3"],
                           "dynamic", 0.12, 11),
            
            # Grünfeld Defence (Volume D)
            OpeningTemplate("Grünfeld Defence Exchange", "D85",
                           ["d4", "Nf6", "c4", "g6", "Nc3", "d5", "cxd5", "Nxd5", "e4", "Nxc3", "bxc3"],
                           "dynamic", 0.14, 11),
            
            OpeningTemplate("Grünfeld Defence Russian System", "D90",
                           ["d4", "Nf6", "c4", "g6", "Nc3", "d5", "Nf3", "Bg7", "Qb3", "dxc4", "Qxc4"],
                           "dynamic", 0.10, 11),
            
            # Benoni Defence (Volume A)
            OpeningTemplate("Modern Benoni", "A70",
                           ["d4", "Nf6", "c4", "c5", "d5", "e6", "Nc3", "exd5", "cxd5", "d6", "Nf3"],
                           "dynamic", 0.11, 11),
            
            OpeningTemplate("Benoni Defence", "A60",
                           ["d4", "Nf6", "c4", "c5", "d5", "e6", "Nc3", "exd5", "cxd5", "d6", "e4"],
                           "dynamic", 0.08, 11),
            
            # Dutch Defence (Volume A)
            OpeningTemplate("Dutch Defence Leningrad", "A80",
                           ["d4", "f5", "g3", "Nf6", "Bg2", "g6", "Nf3", "Bg7", "O-O", "O-O", "c4"],
                           "dynamic", 0.09, 11),
            
            OpeningTemplate("Dutch Defence Stonewall", "A90",
                           ["d4", "f5", "g3", "e6", "Bg2", "Nf6", "Nf3", "Be7", "O-O", "O-O", "c4"],
                           "dynamic", 0.07, 11),
            
            # Pirc Defence (Volume B)
            OpeningTemplate("Pirc Defence", "B07",
                           ["e4", "d6", "d4", "Nf6", "Nc3", "g6", "f4", "Bg7", "Nf3", "c5", "Bb5+"],
                           "dynamic", 0.08, 11),
            
            # Modern Defence (Volume B)
            OpeningTemplate("Modern Defence", "B06",
                           ["e4", "g6", "d4", "Bg7", "Nc3", "d6", "f4", "Nf6", "Nf3", "O-O", "Bd3"],
                           "dynamic", 0.06, 11),
            
            # English Opening dynamic lines (Volume A)
            OpeningTemplate("English Opening Reversed Dragon", "A20",
                           ["c4", "e5", "g3", "h5", "Bg2", "h4", "d3", "Nc6", "Nf3", "f5", "Nc3"],
                           "dynamic", 0.05, 11)
        ]

    def _validate_all_openings(self):
        """Validate that all opening move sequences are legal."""
        total_openings = 0
        valid_openings = 0
        
        for style, openings in self.openings_by_style.items():
            for opening in openings:
                total_openings += 1
                if self._validate_opening_moves(opening.moves):
                    valid_openings += 1
                else:
                    print(f"Invalid opening: {opening.name} - {opening.moves}")
        
        print(f"Validated {valid_openings}/{total_openings} openings")
    
    def _validate_opening_moves(self, moves: List[str]) -> bool:
        """Validate that a sequence of moves is legal."""
        try:
            board = chess.Board()
            for move_str in moves:
                move = board.parse_san(move_str)
                if move not in board.legal_moves:
                    return False
                board.push(move)
            return True
        except:
            return False
    
    def get_openings_for_style(self, style: str) -> List[OpeningTemplate]:
        """Get all openings for a specific style."""
        if style not in self.openings_by_style:
            raise ValueError(f"Unknown style: {style}")
        
        return self.openings_by_style.get(style, [])
    
    def sample_opening_for_style(self, style: str) -> Optional[OpeningTemplate]:
        """Sample a random opening weighted by frequency."""
        openings = self.get_openings_for_style(style)
        if not openings:
            return None
        
        weights = [opening.frequency_weight for opening in openings]
        return np.random.choice(openings, p=np.array(weights) / sum(weights))
    
# Factory function to create the ECO database
# This allows for easy instantiation and testing of the database
# without needing to instantiate the class directly.
def create_eco_opening_database() -> ECO_OpeningDatabase:
    """Factory function to create the ECO opening database."""
    return ECO_OpeningDatabase()