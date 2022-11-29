# Version 24 (commited into Github 29 Nov 2022)
# 
# Play Monopoly and/or Prosperity.
# See the game_rules, sufficient_wealth, setup_game and play_game variables/functions for options.
# As-is, this code:
# * Starts as per normal games with all players having $1500 but no properties.
# * Plays Monopoly until the Gini coefficient exceeds 0.95 and then switches over to Prosperity and runs until all players have at least $3000.
# * Uses abbreviated London site names.


#############################################
# Monopoly and Prosperity
#############################################

import numpy as np
import random


#############################################
# The Game
#############################################

# Can play either game...
game_rules = 'monopoly'
#game_rules = 'prosperity'
# Causes of finishing Monopoly...
turn = 0
gini_index = 0.0
# Causes of finishing Prosperity...
sufficient_wealth = 1500 * 2 # Twice the starting wealth (for all)

#############################################
# Output reporting control
#############################################

# Verbosity flags to aid debugging
RPT_TURN     = 1
RPT_BUY_SELL = 2
RPT_HOUSE    = 4
RPT_RENT     = 16
RPT_STATUS   = 32
RPT_CARD     = 64 # Chance / Community Chest
RPT_MONOPOLY = 128
RPT_DOUBLE   = 256
RPT_WEALTH   = 65536 # bit 17
RPT_EXPECT   = 65536*2
RPT_UNAFFORDABLE = 65536*4
# Edit to control what gets reported e.g.
report = RPT_TURN + RPT_BUY_SELL + RPT_HOUSE + RPT_RENT + RPT_STATUS + RPT_CARD + RPT_MONOPOLY + RPT_DOUBLE
report = RPT_STATUS # Concise normal information provided


#############################################
# Players in the Game
#############################################

# players and player number
NUM_PLAYERS = 4
p_name = ['Aaron', 'Brian', 'Calum', 'David'] # playing in that order

p_state = np.zeros(NUM_PLAYERS, dtype=int)
IS_FREE  = 0
IN_JAIL  = 1
IN_WORKHOUSE  = 2

p_locn    = np.zeros(NUM_PLAYERS, dtype=int)
p_cash    = np.zeros(NUM_PLAYERS, dtype=int)
p_wealth  = np.zeros(NUM_PLAYERS, dtype=int)


#############################################
# Sites on the board
#############################################

# Each site is of a particular type/group...
BROWN   = 0
CYAN    = 1
PINK    = 2
ORANGE  = 3
RED     = 4
YELLOW  = 5
GREEN   = 6
BLUE    = 7
UTILITY = 8
STATION = 9
GO      = 10
TAX     = 11
CHANCE  = 12
CHEST   = 13
GOTOJAIL    = 14
JAIL        = 15
WORKHOUSE   = 16

NUM_SITES = 40

# All the squares clockwise around the board.
site_group = [ GO,       0, CHEST,    0, TAX, STATION, 1, CHANCE,       1, 1,
              JAIL,      2, UTILITY,  2,   2, STATION, 3,  CHEST,       3, 3,
              WORKHOUSE, 4, CHANCE,   4,   4, STATION, 5,      5, UTILITY, 5,
              GOTOJAIL,  6,    6, CHEST,   6, STATION, CHANCE, 7,     TAX, 7 ]

title_price  = [ 0,  60,   0,  60,   0, 200, 100,   0, 100, 120,
                0, 140, 150, 140, 160, 200, 180,   0, 180, 200,
                0, 220,   0, 220, 240, 200, 260, 260, 150, 280,
                0, 300, 300,   0, 320, 200,   0, 350,   0, 400 ]
NOT_FOR_SALE = 0


# Edit to choose between Atlantic City and London names and between full and abbreviated site names
use_names='London'
#use_names='Atlantic City'
use_abbreviations=True

#London sites
if use_names=='London':
    if use_abbreviations:
        # Override: Abbreviated makes reports shorter and easier to view
        site_name = [     'Go..', 'OldK', 'cmch', 'Wtpl', 'InTx', 'KxSt', 'TheA', 'chnc', 'Eust', 'Pent',
                          'Jail', 'Pall', 'Elec', 'Whll', 'Nrth', 'MySt', 'BowS', 'cmch', 'Marl', 'Vine',
                          'Work', 'Strd', 'chnc', 'Flet', 'Traf', 'FnSt', 'Leic', 'Cvty', 'Wter', 'Picc',
                          'GoTo', 'Rege', 'Oxfd', 'cmch', 'Bond', 'LpSt', 'chnc', 'PkLn', 'SpTx', 'Mayf' ]
    else:
        site_name = [       'Go',   'OldKent',    'Chest', 'Whitechapel', 'IncomeTax', 'KingsCross',  'TheAngel',   'Chance', 'EustonRd', 'Pentonville',
                          'Jail',  'PallMall', 'Electric',   'Whitehall',  'Northumb', 'Marylebone', 'BowStreet',    'Chest', 'Marlboro',      'VineSt',
                     'WorkHouse', 'TheStrand',   'Chance',     'FleetSt', 'Trafalgar',  'Fenchurch', 'Leicester', 'Coventry',    'Water',  'Piccadilly',
                      'GoToJail',  'RegentSt', 'OxfordSt',       'Chest',    'BondSt',  'Liverpool',    'Chance', 'ParkLane', 'SuperTax',     'Mayfair' ]
elif use_names=='Atlantic City':
    # Note on South Carolina Ave: https://monopoly.fandom.com/wiki/North_Carolina_Avenue. Similarly on Marvin Gardens.
    if use_abbreviations:
        # Override: Abbreviated makes reports shorter and easier to view
        site_name = [ 'Go..', 'Medi',  'cmch', 'Balt', 'InTx', 'RdgR', 'Ornt', 'chnc', 'Verm', 'Conn',
                      'Jail', 'StCh',  'Elec', 'Stat', 'Virg', 'PnnR', 'St.J', 'cmch', 'Tenn', 'N.Y.',
                      'Poor', 'Kent',  'chnc', 'Indi', 'Illi', 'B&OR', 'Atln', 'Vent', 'Wter', 'Marv',
                      'Blue', 'Pcif',  'Caro', 'cmch', 'Penn', 'ShtR', 'chnc', 'PkPl', 'SpTx', 'Bdwk' ]
    else:
        site_name = [         'Go', 'Mediterranean Ave', 'Community Chest', 'Baltic Ave',      'Income Tax',       'Reading RR',      'Oriental Ave',    'Chance',          'Vermont Ave',   'Connecticut Ave',
                            'Jail', 'St. Charles Place', 'Electric Co',     'States Ave',      'Virginia Ave',     'Pennsylvania RR', 'St. James Place', 'Community Chest', 'Tennessee Ave', 'New York Ave',
                      'Poor House', 'Kentucky Ave',      'Chance',          'Indiana Ave',     'Illinois Ave',     'B&O RR',          'Atlantic Ave',    'Ventnor Ave',     'Water Works',   'Marven Gardens',
                    'Blueblood\'s', 'Pacific Ave',       'S. Carolina Ave', 'Community Chest', 'Pennsylvania Ave', 'Short Line RR',   'Chance',          'Park Place',      'Super Tax',     'Boardwalk Ave' ]
else:
    # Generic: Descriptive of groups
    site_name = [     'Go...', 'Bron1', 'cm.ch', 'Bron2', 'InTx.', 'Stat1', 'Cyan1', 'chnce', 'Cyan2', 'Cyan3',
                      'Jail.', 'Pink1', 'Util1', 'Pink2', 'Pink3', 'Stat2', 'Oran1', 'cm.ch', 'Oran2', 'Oran3',
                      'Work.', 'Red1.', 'chnce', 'Red2.', 'Red3.', 'Stat3', 'Yelo1', 'Yelo2', 'Util2', 'Yelo3',
                      'GoTo.', 'Gren1', 'Gren2', 'cm.ch', 'Gren3', 'Stat4', 'chnce', 'Blue1', 'SpTx.', 'Blue2' ]



group_name = [ 'BROWN', 'CYAN', 'PINK', 'ORANGE', 'RED', 'YELLOW', 'GREEN', 'BLUE', 'UTILITY', 'STATION', 'GO', 'TAX', 'CHANCE', 'CHEST', 'GOTOJAIL', 'JAIL', 'WORKHOUSE' ]

GO_LOCATION        = 0
JAIL_LOCATION      = 10
WORKHOUSE_LOCATION = 20



#############################################
# Ownership
#############################################


NO_PRICE = -1
NO_OWNER = -1
    
# Ownership of property is handled by these arrays...
site_owner = np.zeros(NUM_SITES, dtype=int)
# Rather than counting the number of properties of each group every time it is required, we maintain a count...
site_count_ownership = np.zeros((NUM_PLAYERS, 10), dtype=int)


#############################################
# Money
#############################################

community_chest = 0

def credit(player, price, info):
    p_cash[player]  += price
    #if (player == 0): print('# %5d Credit(%d)+=%d (%s)' % (p_cash[player], player, price, info))

def debit(player, price, info):
    p_cash[player]  -= price
    if (p_cash[player] < 0):
        for locn in reversed(range(NUM_SITES)): # Sell the most-expensive properties first
            if (p_cash[player] < 0) and (site_owner[locn] == player):
                if site_improvements[locn] > 0:
                    site_improvements[locn] = 0 # Sell the house that was there
                    group_improvements[site_group[locn]] -= 1
                if (report & RPT_MONOPOLY) != 0: print('INFO: %s is selling off %s to pay debts!' % (p_name[player], site_name[locn]))
                sell_property(player, locn, title_price[locn]) # And sell back to the bank (no mortgaging)
    if (p_cash[player] < 0):
        if (report & RPT_MONOPOLY) != 0: print('INFO: Bankrupt!: %s has sold off any property but is still in debt!' % (p_name[player]))
        if game_rules == 'monopoly':
            p_locn[player] = WORKHOUSE_LOCATION
            p_state[player] = IN_WORKHOUSE
            p_cash[player]    = 0 # Wipe out any remaining cash (negative values cause an incorrect Gini index calculation)
            # ...and wait until Prosperity rules are invoked

#############################################
# Ownership, Buying and Selling
#############################################

def sell_property(player, location, price):
    global site_count_ownership, site_owner
    if (report & RPT_BUY_SELL) != 0: print('# INFO: %s is selling %s for $%d' % (p_name[player], site_name[location], price))
    if site_owner[location] != player:
        print('ERROR: Property %s belongs to %s hence is not %s\'s to sell!' % (site_name[location], p_name[site_owner[location]], p_name[player]))
    else:
        site_owner[location] = NO_OWNER
        site_count_ownership[player, site_group[location]] -= 1
        credit(player, price, 'sell')
    
def buy_property(player, location, price):
    global site_count_ownership, site_owner
    if site_owner[location] != NO_OWNER:
        # Should already have checked this...
        print('ERROR: %s has already been bought by %s!' % (site_name[location], p_name[site_owner[location]]))
        if (report & RPT_BUY_SELL) != 0: print(site_owner)
        return False
    elif p_cash[player] < price:
        # Should already have checked this...
        print('ERROR: $%d is not enough money for %s to buy %s!' % (p_cash[player], p_name[player], site_name[location]))
        return False
    else:
        site_owner[location] = player
        if (game_rules == 'prosperity') and (site_count_ownership[player, site_group[location]] > 0):
            print('# INFO: %s trying to buy %s for $%d' % (p_name[player], site_name[p_locn[player]], title_price[p_locn[player]]))
            print('# ERROR: already has a property in this color group!')
            print(site_count_ownership)
            exit()
        site_count_ownership[player, site_group[location]] += 1
        debit(player, price, 'buy')
        if (report & RPT_BUY_SELL) != 0:
            print('# INFO: %s buys %s for $%d' % (p_name[player], site_name[p_locn[player]], title_price[p_locn[player]]))
            print('# INFO: %s now has %d properties' % (p_name[player], sum(site_count_ownership[player])))
        return True

def buy_everything(player):
    if (report & RPT_BUY_SELL) != 0: print('# INFO: %s buying everything' % (p_name[player]))
    for location in range(NUM_SITES):
        if title_price[location] != NOT_FOR_SALE :
            buy_property(player, location, title_price[location])
        

#############################################
# Improvements
#############################################

# For each group BROWN...BLUE + UTILITIES + STATIONS
improvement_price = [ 50, 50, 100, 100, 150, 150, 200, 200, NO_PRICE, NO_PRICE ]

def improvable_location(location):
    return (site_group[location] >= BROWN) and (site_group[location] <= BLUE)

site_improvements = np.zeros(NUM_SITES, dtype=int) # Records whether a house has been built on the site
group_improvements = np.zeros(10, dtype=int) # Determines land rent

def make_improvement(player, location):
    if (report & RPT_HOUSE) != 0: print('# INFO: %s is making an improvement on %s' % (p_name[player], site_name[location]))
    if site_owner[location] != player:
        print('ERROR: %s belongs to %s hence is not yours to improve!' % (site_name[location], p_name[site_owner[location]]))
        return False
    elif site_improvements[location] != 0:
        print('ERROR: %s has already been improved!' % (site_name[location]))
        return False
    elif improvement_price[site_group[location]] > p_cash[player]:
        print('ERROR: $%d is not enough money to improve %s!' % (p_cash[player], site_name[location]))
        return False
    elif improvable_location(location): # OK...
        price = improvement_price[site_group[location]]
        debit(player, price, 'improvement')
        site_improvements[location] = 1
        group_improvements[site_group[location]] += 1
        return True
    else:
        print('ERROR: Cannot improve %s!' % (site_name[location]))
        return False

    
def improve_on_everything(player):
    if (report & RPT_HOUSE) != 0: print('# INFO: %s is improving every site owned' % (p_name[player]))
    for location in range(NUM_SITES):
        if improvable_location(location):
            make_improvement(player, location)



#############################################
# Land Rent and Housing Rent
#############################################

# Indices:
UNIMPROVED = 0
# 2 and 3 house prices are used when monopolies are built.
# 4 houses and the hotel price are just provided for completeness
# -1 represents an invalid house price
# none   1    2     3     4   hotel  # site name
site_base_rent = [
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Go
 [  2,  10,  30,   90,  160,  250 ], # Mediterranean Avenue/Old Kent Rd
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Community Chest
 [  4,  20,  60,  180,  320,  450 ], # Baltic Avenue/Whitechapel Rd
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Income Tax
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Reading RR/King's Cross
 [  6,  30,  90,  270,  400,  550 ], # Oriental Avenue/The Angel, Islington
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Chance
 [  6,  30,  90,  270,  400,  550 ], # Vermont Avenue/Euston Rd
 [  8,  40, 100,  300,  450,  600 ], # Connecticut Avenue/Pentonville Rd
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Jail
 [ 10,  50, 150,  450,  625,  750 ], # St  Charles Place/Pall Mall
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Electric Company
 [ 10,  50, 150,  450,  625,  750 ], # States Avenue/Whitehall
 [ 12,  60, 180,  500,  700,  900 ], # Virginia Avenue/Northumberland Ave
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Pensylvania RR/Marylebone
 [ 14,  70, 200,  550,  750,  950 ], # St  James Place/Bow St
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Community Chest
 [ 14,  70, 200,  550,  750,  950 ], # Tennessee Avenue/Marlborough St
 [ 16,  80, 220,  600,  800, 1000 ], # New York Avenue/Vine St
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Just Parking
 [ 18,  90, 250,  700,  875, 1050 ], # Kentucky Avenue/Strand
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Chance
 [ 18,  90, 250,  700,  875, 1050 ], # Indiana Avenue/Fleet St
 [ 20, 100, 300,  750,  925, 1100 ], # Illinois Avenue/Trafalgar Sq
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # B&O RR/Fenchurch St
 [ 22, 110, 330,  800,  975, 1150 ], # Atlantic Avenue/Leicester Sq
 [ 22, 110, 330,  800,  975, 1150 ], # Ventnor Avenue/Coventry St
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Water Works
 [ 24, 120, 360,  850, 1025, 1200 ], # Marven Gardens/Piccadilly
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Go To Jail
 [ 26, 130, 390,  900, 1100, 1275 ], # Pacific Avenue/Regent St
 [ 26, 130, 390,  900, 1100, 1275 ], # South Carolina Avenue/Oxford St
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Community Chest
 [ 28, 150, 450, 1000, 1200, 1400 ], # Pennsylvania Avenue/Bond St
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Short Line RR/Liverpool St
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Chance
 [ 35, 175, 500, 1100, 1300, 1500 ], # Park Place/Park Lane
 [ -1,  -1,  -1,   -1,   -1,   -1 ], # Super Tax
 [ 50, 200, 600, 1400, 1700, 2000 ]] # Boardwalk/Mayfair

num_sites_in_group     = [ 2, 3, 3, 3, 3, 3, 3, 3, 2, 4 ]

# How many properties of the same group does the owner of the location have?
def num_sites_in_group_with_the_same_owner_as(location):
    global site_owner, site_count_ownership, site_group
    if site_owner[location] == NO_OWNER:
        return 0
    else:
        return site_count_ownership[site_owner[location], site_group[location]]

# Does the owner of 'location' own any other sites in the same group?
def there_is_a_monopoly(location):
    global game_rules, site_owner, num_sites_in_group, site_group
    result = False
    if site_owner[location] == NO_OWNER:
        result = False
    else:
        the_site_group = site_group[location]
        num = num_sites_in_group_with_the_same_owner_as(location)
        
        if (the_site_group >= BROWN) and (the_site_group <= STATION):
            if game_rules == 'prosperity':
                # No player is allowed to own more than 1 site in a group
                result = (num > 1)
            else:
                # Notify when a player owns all sites of the same group
                result = (num == num_sites_in_group[the_site_group])
        else:
            print('ERROR: No monopoly is possible on %s!' % (site_name[location]))
            exit()
    if (result == True) and (game_rules == 'prosperity'): print('DEBUG: There is a monopoly on %s owned by %s!' % (site_name[location], p_name[site_owner[location]]))
    return result

def land_rent(location, player):
    if game_rules == 'prosperity':
         # Monopolies are broken. No multiplier effect.
        return site_base_rent[location][UNIMPROVED]
    elif (player == site_owner[location]):
        # Don't pay Land rent on own property in Monopoly
        return 0
    elif site_group[location] == UTILITY:
        # Don't bother with calculating based on die roll; just use average value 7
        if site_count_ownership[player, UTILITY] == 2:
            return 7 * 10 # If both sites owned, 10 times the roll of the dice
        else:
            return 7 * 4  # If one site owned, 4 times the roll of the dice
    elif (site_group[location] >= BROWN) and (site_group[location] <= STATION):
        # The land rent is increased based on the number of improved sites in the group
        # A site owner will benefit from others making improvements within the same group
        if site_improvements[location] == 0:
            # Increase based on the number of improvements based on OTHER sites in the group
            factor_to_be_applied = 1 + group_improvements[site_group[location]]
        else:
            # Increase based on the number of improvements on THIS and other sites in the group
            factor_to_be_applied = group_improvements[site_group[location]]
        return site_base_rent[location][UNIMPROVED] * factor_to_be_applied
    else:
        return 0

def housing_rent(location, player):
    # Monopoly: if the owner of the site owns 2 other sites in the same group,
    #           the rent is according to the classic Monopoly rules 'if 3 houses...'
    # Prosperity: the 'if 1 house...' rule always applies
    if (site_improvements[location]>0) and (player != site_owner[location]):
        if game_rules == 'monopoly':
            factor_to_be_applied = num_sites_in_group_with_the_same_owner_as(location)
            # (This should really be the number of *improved* sites.)
        else:
            factor_to_be_applied = 1 # Monopolies are broken
        if (report & RPT_EXPECT) != 0 and player==0: print('factor ', factor_to_be_applied)
        return site_base_rent[location][factor_to_be_applied]
    else: # No housing_rent chargeable
        return 0


#############################################
# Tracking divergence from equality
#############################################

def next_in_turn(player):
    return (player + 1)%NUM_PLAYERS

def report_wealth(player):
    global title_price
    running_total = p_cash[player]
    for locn in range(NUM_SITES):
        if site_owner[locn] == player:
            LP = title_price[locn]
            running_total += LP
    if (report & RPT_WEALTH) != 0: print('# report_wealth(%d): total = %d' % (player, running_total))
    return running_total
       
def total_outgoings(player):
    # The amount a player x would have to pay if they landed on every square
    running_total = 0
    for locn in range(NUM_SITES):
        if (site_owner[locn] != player):
            running_total += land_rent(locn, player) + housing_rent(locn, player)
            if (report & RPT_EXPECT) != 0 and player==0: print('#outgoings for %s at %12s = %d land & %d housing = %d so far' %(p_name[player], site_name[locn], land_rent(locn, player), housing_rent(locn, player), running_total))
        elif game_rules == 'prosperity':
            running_total += land_rent(locn, player) # Still have to pay land rent on your own site
            if (report & RPT_EXPECT) != 0 and player==0: print('#outgoings for %s at %12s = %d land = %d so far' %(p_name[player], site_name[locn], land_rent(locn, player), running_total))
    return running_total

# Calculate the Gini coefficient which is the ratio of
#   the area under the Lorenz curve (cumulative distribution)
# divided by
#   the area under a 'fair' Lorenz curve.
# Code is based on:
#   https://planspace.org/2013/06/21/how-to-calculate-gini-coefficient-from-raw-data-in-python/
# A more Pythonic alternative is at:
#   https://stackoverflow.com/questions/39512260/calculating-gini-coefficient-in-python-numpy
# Modification to work better with small sample size N:
#   I am multiplying by N/(N-1) so that 'completely unfair' e.g. (0, 0, 0, 1) produces gini(x)=1 rather than (N-1)/N.
def gini_coefficient(list_of_values):
    num_samples = len(list_of_values)
    height, area = 0, 0
    for value in sorted(list_of_values):
        height += value
        area += height - value / 2.
    fair_area = height * num_samples / 2.
    result = ((fair_area - area) / fair_area)*num_samples/(num_samples-1)
    return result


#############################################
# Game turn
#############################################

def setup_game(play='fair'):
    turn = 0
    community_chest = 0
    site_owner[0:]      = [NO_OWNER] * NUM_SITES
    p_locn[0:] = [GO_LOCATION] * NUM_PLAYERS
    if play == 'very_unfair':
        BIG_LANDLORD = 0 # Player #0
        p_cash[0:]    = [1500] * NUM_PLAYERS
        p_cash[BIG_LANDLORD] = 10000
        buy_everything(BIG_LANDLORD)
        improve_on_everything(BIG_LANDLORD)
    else:
        p_cash[0:]    = [1500] * NUM_PLAYERS


def report_status():
    global gini_index
    # shows player info: cash and location, plus the cash in the community chest.
    gini_index = gini_coefficient(p_wealth)
    print('# STATUS: %d G=%.2f ' % (turn, gini_index), end='')
    for plyr in range(NUM_PLAYERS):
        print('(W%5d ' % (p_wealth[plyr]), end='')
        print( 'C%5d ' % (p_cash[plyr]), end='')
        print(  '%s)' % (site_name[p_locn[plyr]]), end='')
    print(' $%d' % (community_chest))

def summarize_sites():
    print('# SITES: ', end='')
    for l in range(NUM_SITES):
        if (site_group[l]>STATION):
            print('.', end='') # Can't buy
        elif (site_owner[l] == NO_OWNER):
            print('-', end='') # Not bought
        elif (site_improvements[l] > 0): # Has a house on it
            name = p_name[site_owner[l]]
            print('%s' % (name[0].upper()), end='') # Upper case initial
        else: # No house
            name = p_name[site_owner[l]]
            print('%s' % (name[0].lower()), end='') # Lower case initial
    print('')


def i_can_afford_it(player, price): # Based on my cash being higher than that required in the next few rounds!
    return price < (p_cash[player] - total_outgoings(player))


def game_turn(player, dice_throw):
    global p_locn
    global p_cash
    global p_wealth
    global site_owner
    global site_count_ownership
    global site_group
    global site_improvements
    global community_chest
    global report
    global turn

    # 1. Stuck in the Workhouse
    # 2. Stuck in Jail
    # 3. Collecting $200 for passing Go (possibly building houses)
    if (p_state[player] == IN_WORKHOUSE) and (p_cash[player] <= 0):
        # To leave the workhouse, you must have some money.
        # Having lost everything, you will only get money via the Community Chest
        pass
    elif (p_state[player] == IN_JAIL) and not threw_a_double(dice_throw):
        # In this program: the only way out of jail is to (eventually) throw a double
        pass
    else:
        p_state[player] = IS_FREE
        p_locn[player] = (p_locn[player] + dice_throw[0] + dice_throw[1])
        if p_locn[player] >= NUM_SITES:
            p_locn[player] -= NUM_SITES # Module 40 position
            credit(player, 200, 'go')
            if (report & RPT_TURN) != 0: print('# INFO: %s collecting $200 for passing GO' % (p_name[player]))
            # Allowed to build houses when passing GO...
            for location in range(NUM_SITES):
                # Put houses on cheapest properties first (gradually build up)
                if (site_owner[location] == player) and improvable_location(location):
                    if not (site_improvements[location]) and i_can_afford_it(player, improvement_price[site_group[location]]):
                        make_improvement(player, location)
        if (report & RPT_TURN) != 0: print('# INFO: %s rolled %d and is now at %s (%s)' % (p_name[player], dice_throw[0]+dice_throw[1], site_name[p_locn[player]], group_name[site_group[p_locn[player]]]))

    # What to do depends on what square you have ended up on...
    if site_group[p_locn[player]]== GO:
        if (report & RPT_TURN) != 0: print('# INFO: %s is at GO' % (p_name[player]))
        pass # Will have already collected $200
    elif site_group[p_locn[player]]== WORKHOUSE:
        if (p_state[player] == IS_FREE):
            if (report & RPT_TURN) != 0: print('# INFO: %s is just visiting the Workhouse' % (p_name[player]))
    elif site_group[p_locn[player]]== JAIL: # Now 'Just Visiting'
        if (p_state[player] == IS_FREE):
            if (report & RPT_TURN) != 0: print('# INFO: %s is just visiting jail' % (p_name[player]))
    elif site_group[p_locn[player]]== GOTOJAIL:
        if game_rules == 'monopoly':
            if (report & RPT_TURN) != 0: print('# INFO: %s goes to jail for trespassing on Lord Blueblood\'s land!' % (p_name[player]))
            p_locn[player]  = JAIL_LOCATION
            p_state[player] = IN_JAIL
        else:
            if (report & RPT_TURN) != 0: print('# INFO: %s is just visiting Blueblood\'s park' % (p_name[player]))
        pass
    elif site_group[p_locn[player]]== TAX:
        if game_rules == 'prosperity':
            if (report & RPT_TURN) != 0: print('# INFO: %s does not need to pay any taxes other than Land Value Tax!' % (p_name[player]))
        elif site_name[p_locn[player]] == 'IncomeTax':
            if (report & RPT_TURN) != 0: print('# INFO: %s is paying $200 income tax' % (p_name[player]))
            debit(player, 200, 'income tax')
        else:
            if (report & RPT_TURN) != 0: print('# INFO: %s is paying $100 super tax' % (p_name[player]))
            debit(player, 100, 'super tax')
    elif site_group[p_locn[player]]== CHANCE: # Now 'Absolute Necessity' $10
        if (report & RPT_TURN) != 0:
            if(game_rules == 'prosperity'):
                if (p_wealth[player] > 10):
                    if (report & RPT_CARD) != 0: print('# INFO: %s is paying $10 for an Absolute Necessity' % (p_name[player]))
                    debit(player, 10, 'chance')
                    community_chest      += 10
            else:
                # Local rules: have the option to take a Chance card.
                # In this automated game, always choose not to.
                print('# INFO: %s has decided not to take a chance' % (p_name[player]))
    elif site_group[p_locn[player]]== CHEST: # Community Chest
        if (game_rules == 'monopoly'):
            # Local rules: have the option to take a Community Chest card.
            # In this automated game, always choose not to.
            if (report & RPT_TURN) != 0: print('# INFO: %s does not get involved with the Community Chest' % (p_name[player]))
        else: # prosperity
            if community_chest > NUM_PLAYERS: # Sharing out: At least $1 to distribute!
                if (report & RPT_CARD) != 0: print('# Distributing the Community Chest')
                fee = community_chest // NUM_PLAYERS # (floor division)
                for plyr in range(NUM_PLAYERS):
                    if (report & RPT_CARD) != 0: print('# INFO: %s is receiving $%d from the Community Chest' % (p_name[plyr], fee))
                    credit(plyr, fee, 'payout')
                    community_chest      -= fee
    else: # A group site (housing color, station or utility)
        ###print('DEBUG_POINT3')
        #################################
        # Should there be a forced sale? (to break a monopoly)
        #################################
        ##if there_is_a_monopoly(p_locn[player]):
        #if (game_rules == 'prosperity'):
        #    #if num_sites_in_group_with_the_same_owner_as(p_locn[player]) > 1:
        #    if site_count_ownership[site_owner[p_locn[player]], site_group[p_locn[player]]] > 1:
        #        print('DEBUG_POINT3 AT %s' % (site_name[p_locn[player]]))
        #######        print(site_count_ownership)
        #if num_sites_in_group_with_the_same_owner_as(p_locn[player]) > 1:
        if site_count_ownership[site_owner[p_locn[player]], site_group[p_locn[player]]] > 1:
            if (game_rules == 'prosperity') and (player != site_owner[p_locn[player]]):
                # Monopoly only broken if the player landing on the site is not the monopoly owner!
                ######summarize_sites()
                if (report & RPT_BUY_SELL) != 0:
                    print('# INFO: %s is forced to sell %s to break monopoly' % (p_name[site_owner[p_locn[player]]], site_name[p_locn[player]]))
                    print(site_count_ownership)
                # Reimburse for any improvement:
                if site_improvements[p_locn[player]] == 1:
                    site_improvements[p_locn[player]] = 0
                    credit(site_owner[p_locn[player]], improvement_price[site_group[p_locn[player]]], 'reimbursement')
                sell_property(site_owner[p_locn[player]], p_locn[player], title_price[p_locn[player]])
        #################################
        # Will the site be bought?
        #################################
        ###print('DEBUG_POINT4')
        ###print(site_owner[p_locn[player]])
        ####print(title_price[p_locn[player]])
        if site_owner[p_locn[player]] == NO_OWNER:
            ###print('DEBUG_POINT3')
            price_to_buy = title_price[p_locn[player]]
            if i_can_afford_it(player, price_to_buy):
                ##########if (report & RPT_BUY_SELL) != 0: print('# INFO: %s is buying %s for $%d' % (p_name[player], site_name[p_locn[player]], price_to_buy))
                count = site_count_ownership[player, site_group[p_locn[player]]]
                #if (game_rules == 'prosperity') and there_is_a_monopoly(p_locn[player]):
                if (game_rules == 'prosperity') and (count > 0):
                    if (report & RPT_BUY_SELL) != 0: print('# INFO: %s not allowed to buy %s' % (p_name[player], site_name[p_locn[player]]))
                    ####print('DEBUG_POINT7')
                else:
                    ######summarize_sites() ##### DEBUG to see why players can buy towards a monopoly...
                    ######print(num_sites_in_group_with_the_same_owner_as(p_locn[player]))
                    #####print(there_is_a_monopoly(p_locn[player]))
                    buy_property(player, p_locn[player], price_to_buy)
                    #####print(site_count_ownership) ##### DEBUG to see why players can buy towards a monopoly...
            else:
                if (report & RPT_UNAFFORDABLE) != 0: print('# INFO: %s cannot afford to buy %s' % (p_name[player], site_name[p_locn[player]]))

        #################################
        # Will rent be paid?
        #################################
        if (player != site_owner[p_locn[player]]):
            if (report & RPT_TURN) != 0: print('# INFO: %s owns %s' % (p_name[site_owner[p_locn[player]]], site_name[p_locn[player]]))
            fee = land_rent(p_locn[player], player)
            if (game_rules == 'prosperity'):
                # Pay Land Rent to the Community Chest (including owners of the site)
                if (report & RPT_RENT) != 0: print('# INFO: %s is paying $%d Land Rent for %s' % (p_name[player], fee, site_name[p_locn[player]]))
                debit(player, fee, 'land rent')
                community_chest      += fee
            elif player != site_owner[p_locn[player]]: # monopoly rules
                # Pay Land Rent to the owner
                if (report & RPT_RENT) != 0: print('# INFO: %s is paying $%d Land Rent to %d for %s' % (p_name[player], fee, site_owner[p_locn[player]], site_name[p_locn[player]]))
                debit(player, fee, 'land rent')
                credit(site_owner[p_locn[player]], fee, 'let land')
                # Pay Housing Rent to the owner if applicable
                if player != site_owner[p_locn[player]]:
                    fee = housing_rent(p_locn[player], player)
                    if (report & RPT_RENT) != 0: print('# INFO: %s is paying $%d House Rent to %d for %s' % (p_name[player], fee, site_owner[p_locn[player]], site_name[p_locn[player]]))
                    debit(player, fee, 'house rent')
                    credit(site_owner[p_locn[player]], fee, 'let house')
        # (Code is not implementing the jump +/-8 rule on stations)

    for player2 in range(NUM_PLAYERS):
        p_wealth[player2] = report_wealth(player2)
        
    # Running status:
    # shows player info: cash, 'gain' and location, plus the cash in the community chest.
    # 'Gain' is an estimate of how quickly cash will increase or (if negative) decrease.
    if ((report & RPT_STATUS) != 0): report_status()
    ######if report_the_status: summarize_sites()

#############################################
# Game play
#############################################

def threw_a_double(dice_throw):
    return dice_throw[0] == dice_throw[1]

def play_game(max_turns, max_gini):
    global p_locn
    global p_cash
    global site_owner
    global site_count_ownership
    global site_improvements
    global turn
    global game_rules
    global report

    print('#####################')
    if (game_rules == 'prosperity'):
        print('# Let\'s play Prosperity!')
    elif (game_rules == 'monopoly'):
        print('# Let\'s play Monopoly!')
    else:
        print('# ERROR: Bad game')
        exit()
    print('#####################')

    turn = 1
    player = 0 # Player numbers are arbitrary. Fixed starting player (not rolling die and seeing who has the highest number)
    finish = False
    while (turn <= max_turns) and not finish:
        if (site_group[p_locn[player]]== WORKHOUSE) and (p_cash[player] < 0):
            if (report & RPT_TURN) != 0: print('# INFO: %s is stuck in the Workhouse' % (p_name[player]))
        else:
            if (report & RPT_TURN) != 0: print('# INFO: %s is at %s at start of turn %d with $%d' % (p_name[player], site_name[p_locn[player]], turn, p_cash[player]))
            dice_throw = [ random.randint(1,6), random.randint(1,6) ]
            game_turn(player, dice_throw)
            if threw_a_double(dice_throw):
                if (report & RPT_DOUBLE) != 0: print('# INFO: Another go because %s threw a double' % (p_name[player]))
                dice_throw = [ random.randint(1,6), random.randint(1,6) ]
                game_turn(player, dice_throw)
                if threw_a_double(dice_throw):
                    if (report & RPT_DOUBLE) != 0: print('# INFO: Another go because %s threw a second double' % (p_name[player]))
                    dice_throw = [ random.randint(1,6), random.randint(1,6) ]
                    game_turn(player, dice_throw)
                    if threw_a_double(dice_throw):
                        if (report & RPT_DOUBLE) != 0: print('# INFO: Go to jail for speeding (%s threw a third double' % (p_name[player]))
                        p_locn[player] = JAIL_LOCATION
                        p_locn[player] = IN_JAIL
                    else:
                        game_turn(player, dice_throw)
        player = next_in_turn(player)
        turn = turn + 1
        if (gini_index >= max_gini):
            finish = True
            print('##################################################')
            print('# Game over!: Gini index %.2f has exceeded %.2f' %(gini_index, max_gini))
            print('##################################################')
        elif (game_rules == 'prosperity'):
            # prosperity: continues until all players have sufficient wealth
            finish = True # until proven otherwise...
            for plyr in range(NUM_PLAYERS):
                if p_wealth[plyr] < sufficient_wealth:
                    finish = False
            if finish:
                print('##################################################')
                print('# Game over!: All players now have more than $%d' % (sufficient_wealth))
                print('##################################################')
        if (turn % 100 == 99) or finish:
            # Every so often, and at the very end, provide a snapshot of ownership
            #print('# Number of sites owned by each player, on each site group BROWN...BLUE + UTILITIES + STATION')
            #for p in range(NUM_PLAYERS):
            #####    print(site_count_ownership[p])
            #for l in range(NUM_SITES):
            #    if site_owner[l] != NO_OWNER:
            #        print('Site %s owned by %s with %d houses' % (site_name[l], p_name[site_owner[l]], site_improvements[l]))
            report_status()
            summarize_sites()
            #print(site_owner)
   

#############################################
# Start of program execution
#############################################

# Deterministic behaviour: The seed for random number generation is not based on time/date:
random.seed(1) # Arbitrary number

if False: # Monopoly only
    setup_game(play='fair')
    game_rules = 'monopoly'
    play_game(max_turns=5000, max_gini=0.99)

if False: # Prosperity only
    report = RPT_STATUS + RPT_BUY_SELL + RPT_HOUSE
    setup_game(play='fair')
    game_rules = 'prosperity'
    play_game(max_turns=500, max_gini=0.40)

if True: # Monopoly then Prosperity
    report = RPT_STATUS + RPT_BUY_SELL + RPT_HOUSE
    setup_game(play='fair')
    game_rules = 'monopoly'
    play_game(max_turns=1000, max_gini=0.95)
    game_rules = 'prosperity'
    play_game(max_turns=1000, max_gini=0.99)

if False: # Monopoly then Prosperity
    setup_game(play='fair')
    game_rules = 'monopoly'
    play_game(max_turns=5000, max_gini=0.95)
    game_rules = 'prosperity'
    sufficient_wealth = 1000000 # Never! Let 'max_turns' stop the sim
    play_game(max_turns=5000, max_gini=0.99)
    
print(site_count_ownership)

print('###################')
print('# End of simulation')
print('###################')
