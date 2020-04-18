import unittest
import nflgame

class TestGame(unittest.TestCase):

    def test_game_can_query_by_week(self):
        games = nflgame.games(2013, week=1)
        for play_no,p in enumerate(plays):
            if(play_no == 1):
                self.assertTrue(play.EPA < 0)


if __name__ == '__main__':
    unittest.main()