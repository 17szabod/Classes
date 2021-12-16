# Code thanks to the MGGG group!
# https://github.com/mggg
import math


def get_percents(counts, totals):
    return {
        part: counts[part] / totals[part] if totals[part] > 0 else math.nan
        for part in totals
    }


# My code to make life a little easier without fully adhering to MGGG's structure:
class Results:
    """
    Represents the results of an election. Provides helpful methods to answer
    common questions you might have about an election (Who won? How many seats?, etc.).
    """

    def __init__(self, counts, races, parties):
        self.parties = parties
        self.totals_for_party = counts
        self.races = races

        self.totals = {
            race: sum(counts[party][race] for party in self.parties)
            for race in self.races
        }

        self.percents_for_party = {
            party: get_percents(counts[party], self.totals)
            for party in self.parties
        }

    def seats(self, party):
        """
        Returns the number of races that ``party`` won.
        """
        return sum(self.won(party, race) for race in self.races)

    def wins(self, party):
        """
        An alias for :meth:`seats`.
        """
        return self.seats(party)

    def percent(self, party, race=None):
        """
        Returns the percentage of the vote that ``party`` received in a given race
        (part of the partition). If ``race`` is omitted, returns the overall vote
        share of ``party``.
        :param party: Party ID.
        :param race: ID of the part of the partition whose votes we want to tally.
        """
        if race is not None:
            return self.percents_for_party[party][race]
        return sum(self.votes(party)) / sum(self.totals[race] for race in self.races)

    def percents(self, party):
        """
        :param party: The party
        :return: The tuple of the percentage of votes that ``party`` received
            in each part of the partition
        """
        return tuple(self.percents_for_party[party][race] for race in self.races)

    def count(self, party, race=None):
        """
        Returns the total number of votes that ``party`` received in a given race
        (part of the partition). If ``race`` is omitted, returns the overall vote
        total of ``party``.
        :param party: Party ID.
        :param race: ID of the part of the partition whose votes we want to tally.
        """
        if race is not None:
            return self.totals_for_party[party][race]
        return sum(self.totals_for_party[party][race] for race in self.races)

    def counts(self, party):
        """
        :param party: Party ID
        :return: tuple of the total votes cast for ``party`` in each part of
            the partition
        """
        return tuple(self.totals_for_party[party][race] for race in self.races)

    def votes(self, party):
        """
        An alias for :meth:`counts`.
        """
        return self.counts(party)

    def won(self, party, race):
        """
        Answers "Did ``party`` win the race in part ``race``?" with ``True`` or ``False``.
        """
        return all(
            self.totals_for_party[party][race] >= self.totals_for_party[opponent][race]
            for opponent in self.parties
        )

    def total_votes(self):
        return sum(self.totals.values())