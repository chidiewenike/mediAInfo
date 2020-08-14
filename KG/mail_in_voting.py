from grakn.client import GraknClient
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

VOTING_METHOD = 1
CRIME = 2
PERPETRATOR = 3
LEADER = 4
POLITY = 5
STAT_VAL = 6

def read_csv(path_to_file):
    """
    Reading the csv with only the columns that we are interested
    and filling the missing data using pandas.
    """
    using_cols = ['entity_name',
                  'entity',
                  'leads',
                  'legal_issue',
                  'allows_for',
                  'case',
                  'rate_of_crime',
                  'country',
                  'state',
                  'president',
                  'governor',
                  'age',
                  'origin',
                  'population',
                  'capital',
                  'citizenship',
                  'gender',
                  'term',
                  'stat_value',
                  'source']

    # specify the columns that we are interested in
    data = pd.read_csv(path_to_file, usecols = using_cols)

    # others will be filled with empty string
    data = data.fillna("")
    return data

def load_data_into_grakn(session,input_df):
    """
    Loading the data form the DataFrame to the graph in parts
    """

    print("Inserting leaders...")
    input_df.progress_apply(insert_one_leader, axis=1, session=session)

    print("Inserting polities...")
    input_df.progress_apply(insert_one_polity, axis=1, session=session)

    print("Inserting voting methods...")
    input_df.progress_apply(insert_one_voting_method, axis=1, session=session)

    print("Inserting criminals...")
    input_df.progress_apply(insert_one_criminal, axis=1, session=session)

    print("Inserting stats...")
    input_df.progress_apply(insert_one_stat, axis=1, session=session)

    print("Inserting crimes...")
    input_df.progress_apply(insert_one_crime, axis=1, session=session)

    print("Inserting leads...")
    input_df.progress_apply(insert_one_leads, axis=1, session=session)

    print("Inserting allows for...")
    input_df.progress_apply(insert_one_allows_for, axis=1, session=session)

    print("Inserting legal issues...")
    input_df.progress_apply(insert_one_legal_issues, axis=1, session=session)

    print("Inserting legal case...")
    input_df.progress_apply(insert_one_case, axis=1, session=session)

    print("Inserting legal case...")
    input_df.progress_apply(insert_one_rate, axis=1, session=session)

def insert_one_leader(df,session):
    """
    Given one row of data, insert one character to the graph.
    """
    if int(df.entity) == LEADER:

        if df.president == 1:
            title = 'president'
        elif df.governor == 1:
            title = 'governor'

        # write the graql query
        graql_insert_query = f'insert $leader isa leader, ' \
                            f'has name "{df.entity_name}", ' \
                            f'has title "{title}", ' \
                            f'has gender "{df.gender}", ' \
                            f'has citizenship "{df.citizenship}", ' \
                            f'has origin "{df.origin}", ' \
                            f'has age {df.age}, ' \
                            f'has gender "{df.gender}", ' \
                            f'has source "{df.source}", ' \
                            f'has term "{df.term}";'

        with session.transaction().write() as transaction:
            # make a write transection with the query
            transaction.query(graql_insert_query)
            # remember to commit at the end
            transaction.commit()

def insert_one_polity(df,session):
    """
    Given one row of data, insert one character to the graph.
    """

    if int(df.entity) == POLITY:
        title = ""

        if df.country == 1:
            title = 'country'
        elif df.state == 1:
            title = 'state'

        # write the graql query
        graql_insert_query = f'insert $polity isa polity, ' \
                            f'has name "{df.entity_name}", ' \
                            f'has title "{title}", ' \
                            f'has population {int(df.population)}, ' \
                            f'has capital "{df.capital}", ' \
                            f'has source "{df.source}"; '

        with session.transaction().write() as transaction:
            # make a write transection with the query
            transaction.query(graql_insert_query)
            # remember to commit at the end
            transaction.commit()

def insert_one_crime(df,session):
    """
    Given one row of data, insert one character to the graph.
    """

    if int(df.entity) == CRIME:
        # write the graql query
        graql_insert_query = f'insert $crime isa crime, ' \
                            f'has name "{df.entity_name}", ' \
                            f'has source "{df.source}"; '

        with session.transaction().write() as transaction:
            # make a write transection with the query
            transaction.query(graql_insert_query)
            # remember to commit at the end
            transaction.commit()

def insert_one_voting_method(df,session):
    """
    Given one row of data, insert one character to the graph.
    """
    if df.entity == VOTING_METHOD:
        # write the graql query
        graql_insert_query = f'insert $voting_method isa voting_method, ' \
                            f'has name "{df.entity_name}", ' \
                            f'has source "{df.source}"; '

        with session.transaction().write() as transaction:
            # make a write transection with the query
            transaction.query(graql_insert_query)
            # remember to commit at the end
            transaction.commit()

def insert_one_criminal(df,session):
    """
    Given one row of data, insert one character to the graph.
    """

    if int(df.entity) == PERPETRATOR:
        # write the graql query
        graql_insert_query = f'insert $criminal isa criminal, ' \
                            f'has name "{df.entity_name}", ' \
                            f'has source "{df.source}"; '

        with session.transaction().write() as transaction:
            # make a write transection with the query
            transaction.query(graql_insert_query)
            # remember to commit at the end
            transaction.commit()

def insert_one_stat(df,session):
    """
    Given one row of data, insert one character to the graph.
    """
    print(df.entity)
    if int(df.entity) == STAT_VAL:

        # write the graql query
        graql_insert_query = f'insert $stat isa stat, ' \
                            f'has name "{df.entity_name}", ' \
                            f'has stat_value "{df.stat_value}", ' \
                            f'has source "{df.source}"; '

        with session.transaction().write() as transaction:
            # make a write transection with the query
            transaction.query(graql_insert_query)
            # remember to commit at the end
            transaction.commit()

def write_to_graph(session, kg_query):
    with session.transaction().write() as transaction:

        # make a write transection with the query
        transaction.query(kg_query)
        # remember to commit at the end
        transaction.commit()    

    return session

def insert_one_allows_for(df,session):
    if df["allows_for"] == "":
        return None

    if (df.allows_for == "Postal"):
        if int(df.country) == 1:
            print("ADDING POSTAL COUNTRY: %s" % df.entity_name)
            # write the graql query
            graql_insert_query = f'match $polity isa polity, ' \
                            f'has name "{df.entity_name}"; ' \
                            f'$postal isa voting_method, ' \
                            f'has name "{df.allows_for}";' \
                            f'insert $allows_for(postal: $postal, country: $polity) isa allows_for;'

        # Country
        else:
            print("ADDING POSTAL STATE: %s" % df.entity_name)
            # write the graql query
            graql_insert_query = f'match $polity isa polity, ' \
                            f'has name "{df.entity_name}"; ' \
                            f'$postal isa voting_method, ' \
                            f'has name "{df.allows_for}";' \
                            f'insert $allows_for(postal: $postal, state: $polity) isa allows_for;'
    session = write_to_graph(session, graql_insert_query)

def insert_one_leads(df,session):
    """
    Given one row of data, insert one marriage to the graph.
    """
    if df['leads'] == "":
        return None

    if int(df.governor) == 1:
        # write the graql query
        graql_insert_query = f'match $leader isa leader, ' \
                        f'has name "{df.entity_name}"; ' \
                        f'$state isa polity, ' \
                        f'has name "{df.leads}";' \
                        f'insert $leads(governor: $leader, state: $state) isa leads;'

    # President
    else:
        # write the graql query
        graql_insert_query = f'match $leader isa leader, ' \
                        f'has name "{df.entity_name}"; ' \
                        f'$country isa polity, ' \
                        f'has name "{df.leads}";' \
                        f'insert $leads(president: $leader, country: $country) isa leads;'

    session = write_to_graph(session, graql_insert_query)

def insert_one_legal_issues(df,session):
    if df["legal_issue"] == "":
        return None

    if (df.legal_issue == "Voting Fraud"):
        graql_insert_query = f'match $voting_method isa voting_method, ' \
                        f'has name "{df.entity_name}"; ' \
                        f'$felony isa crime, ' \
                        f'has name "{df.legal_issue}";' \
                        f'insert $legal_issue(felony: $felony, postal: $voting_method) isa legal_issue;'
    session = write_to_graph(session, graql_insert_query)

def insert_one_case(df,session):
    if df["case"] == "":
        return None

    if (df.case == "Katie Meyer"):
        graql_insert_query = f'match $crime isa crime, ' \
                        f'has name "{df.entity_name}"; ' \
                        f'$perpetrator isa criminal, ' \
                        f'has name "{df.case}";' \
                        f'insert $case(felony: $crime, perpetrator: $perpetrator) isa case;'
    session = write_to_graph(session, graql_insert_query)

def insert_one_rate(df,session):
    if df["rate_of_crime"] == "":
        return None

    if (df.rate_of_crime == "0.0009%"):
        graql_insert_query = f'match $crime isa crime, ' \
                        f'has name "{df.entity_name}"; ' \
                        f'$rate isa stat, ' \
                        f'has name "{df.rate_of_crime}";' \
                        f'insert $rate_of_crime(felony: $crime, rate: $rate) isa rate_of_crime;'
    session = write_to_graph(session, graql_insert_query)

def build_grakn_graph(input_df, keyspace_name):
    """
    Create a connection with the graph with a specifil keyspace
    using the GraknClient and load the DataFrame into the graph
    """
    with GraknClient(uri="localhost:48555") as client:
        with client.session(keyspace = keyspace_name) as session:
            load_data_into_grakn(session,input_df)

# read the csv into DataFrame, it is stored in the sub-directory named 'data'
raw_data = read_csv('voting.csv')
# call the function to build and load the graph from the DataFrame
build_grakn_graph(raw_data, 'mail_in_voting')