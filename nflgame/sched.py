from collections import OrderedDict
import datetime
import json
import os.path

__pdoc__ = {}

_sched_json_file = os.path.join(os.path.dirname(__file__), 'schedule.json')


def calc_desired_weeks(year, phase):
    desired_weeks = []

    for week in range(5):
        desired_weeks.append(tuple([year, 'PRE', week]))
    for week in range(1,18):
        desired_weeks.append(tuple([year,'REG',week]))

    if phase is 'POST':
        for week in range(1,5):
            desired_weeks.append(tuple([year, 'POST', week]))

    return desired_weeks


def check_missing_weeks(sched, year, phase):

    missing_weeks = calc_desired_weeks(year, phase)
    stored_weeks = set()

    for info in sched.values():
        if info['year'] != year:
            continue
        stored_week = (year, info['season_type'], info['week'])
        stored_weeks.add(stored_week)

    for stored_week in stored_weeks:
        missing_weeks.remove(stored_week)

    return missing_weeks


def order_weeks_to_update(missing_weeks, current_week):
    if current_week in missing_weeks:
        missing_weeks.remove(current_week)

    missing_weeks.insert(0, current_week)
    return missing_weeks


def _create_schedule(jsonf=None):
    """
    Returns an ordered dict of schedule data from the schedule.json
    file, where games are ordered by the date and time that they
    started. Keys in the dictionary are GSIS ids and values are
    dictionaries with the following keys: week, month, year, home,
    away, wday, gamekey, season_type, time.
    """
    day = 60 * 60 * 24
    if jsonf is None:
        jsonf = _sched_json_file
    try:
        data = json.loads(open(jsonf).read())
    except IOError:
        return OrderedDict(), datetime.datetime.utcnow()

    sched = OrderedDict()
    for gsis_id, info in data.get('games', []):
        sched[gsis_id] = info
    last_updated = datetime.datetime.utcfromtimestamp(data.get('time', 0))

    if (datetime.datetime.utcnow() - last_updated).total_seconds() >= day:
        # Only try to update if we can write to the schedule file.
        if os.access(jsonf, os.W_OK):
            import nflgame.live
            import nflgame.update_sched
            year, week = nflgame.live.current_year_and_week()
            phase = nflgame.live._cur_season_phase
            current_week = (year, phase, week)

            missing_weeks = check_missing_weeks(sched, year, phase)
            weeks_to_update = order_weeks_to_update(missing_weeks, current_week)

            for week_to_update in weeks_to_update:
                print(('Updating {}').format(week_to_update))
                year, phase, week = week_to_update
                week_was_updated = nflgame.update_sched.update_week(sched, year, phase, week)
                if not week_was_updated:
                    print(("Week {}{} of {} was either empty, or it couldn't be fetched from NFL.com. Aborting.")\
                        .format(phase , week, year))
                    break

            nflgame.update_sched.write_schedule(jsonf, sched)
            last_updated = datetime.datetime.utcnow()

    return sched, last_updated

games, last_updated = _create_schedule()

__pdoc__['nflgame.sched.games'] = """
An ordered dict of schedule data, where games are ordered by the date
and time that they started. Keys in the dictionary are GSIS ids and
values are dictionaries with the following keys: week, month, year,
home, away, wday, gamekey, season_type, time.
"""

__pdoc__['nflgame.sched.last_updated'] = """
A `datetime.datetime` object representing the last time the schedule
was updated.
"""
