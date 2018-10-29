import nflgame

games = nflgame.games(2013, week=1)
plays = nflgame.combine_plays(games)
for p in plays:
    print(p.EPA)
