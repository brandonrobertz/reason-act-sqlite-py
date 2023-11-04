import json
import os
import re
import sys
import sqlite3

from llama_cpp import Llama
import sqlite_utils


DB_PATH = "example.db"

# columns: useful for getting table columns. input 1: table name.
ACTIONS = """
tables: useful for getting the names of tables available. no input.
schema: useful for looking at the schema of a database. input 1: table name.
help: useful for getting helpful context about how to use tables and their columns. input 1: table name. (optional) input 2: column name.
sql-query: useful for analyzing data and getting the top 5 results of a query. input 1: a valid sqlite sql query.
"""

DATA_HELP = {
    "users": {
        None: "profiles of individuals (sometimes called creators) who are seeking work, have worked on projects, or are looking to hire other people.",
        "creatorUserId": "this is the primary key for a user. the experiences table references it on the creatorUserId field",
        "createdUtc": "a ISO8601 datetime string of the user creation date",
        "updatedUtc": "a ISO8601 datetime string of the user's last updated date",
        "isPublic": "a boolean describing if the user's profile is public. all of these values will be true",
        "isContactAllowed": "a boolean describing whether or not the user allows people to contact them",
        "creatorDescription": "a free-text field the user has supplied describing themselves, their interests, work preferences and occasionally age/location. details like this are sometimes present: 'I am 23 years old' or 'been building games for 8 years'.",
        "isOpenToWork": "whether or not the user is actively looking for work",
        "interestDescription": "a text field describing the users interests",
        "linkTypes": "an array of platforms/methods the users can be contacted on",
        "preferredContactLinkType": "the user's preferred platform/method of contact",
        "socialLinks": "an array of JSON data describing the user's social media accounts",
        "jobTypes": "the type of jobs and work the user is seeking",
        "skillTypes": "an array containing skills the user has",
        "requiresAction": "always set to \"noAction\"",
    },
    "experiences": {
        # table description
        None: "the experiences table describes previous jobs that users have worked on.",
        "experienceId": "the primary key for this experience",
        "creatorUserId": "the primary key from the users table of the user who put this experience on this profile",
        "createdUtc": "an ISO8601 experience create date",
        "updatedUtc": "an ISO8601 experience update date",
        "projectName": "the name of the project the user worked on. this matches the name column of the games table.",
        "experienceDescription": "a free-text field describing the work the user performed",
        "jobRole": "a free-text field describing the user's job title on the project",
        "experienceMedia": "an array of JSON objects describing media displayed on the profile. the names of the media may have descriptive meaning",
        "experienceLinks": "an array of text containing Markdown links to the game(s) for this experience. the ID in the games URLs may be present in the games table",
        "teamName": "the name of the team the user worked with, if applicable",
        "teamId": "the team ID, which may be different from the teamName across experiences",
        "startedUtc": "an ISO8601 date of when the user began work on the project",
        "endedUtc": "an ISO8601 date of when the user ended work on the project",
        "isCurrent": "wheter or not the project is ongoing",
    },
    "games": {
        None: "these are games that are available to play. users mention them in their experience descriptions.",
        "placeId": "this is the game's primary key. this could show up as an ID in a games URL elsewhere",
        "name": "the name of the game",
        "description": "a free-form text description of the game",
        "sourceName": "the name of the game",
        "sourceDescription": "a free-form text description of the game",
        "url": "the URL to the game's page. can be found in experienceLinks or user descriptions. the placeId appears in the URL",
        "builder": "the name of the user who built the game",
        "builderId": "the ID of the user who build the game",
        "hasVerifiedBadge": "whether or not the game has gone through the verification process",
        "isPlayable": "whether or not the game is playable",
        "reasonProhibited": "this field, if it's not \"None\", provides the reason why the game isn't currently playable",
        "universeId": "I'm not sure what this field is. this ID may appear in user or experience descriptions",
        "universeRootPlaceId": "I'm not sure what this field is. this ID may appear in user or experience descriptions",
        "price": "the price of this game",
        "imageToken": "an ID representing the game's image",
    },
    "game_stats": {
        None: "game stats record popularity metrics and game categories. the numbers here are all intended to be short and human-readable, not sortable",
        "Active": "how many users are active on the game",
        "Favorites": "how many users have favorited the game",
        "Visits": "how many users have visited or played the game",
        "Created": "when the game was created",
        "Updated": "when the last time the game was updates",
        "Server Size": "how large the server is",
        "Genre": "a category the game falls into",
        "Allowed Gear": "this field is always blank",
        "placeId": "a primary key of the game. this matches a row in the games table",
    },
    "game_passes": {
        None: "game passes are add-ons that players can purchase that grant additional capabilities in games",
        "name": "the title of the add-on. this is not the game's name. that must be found via joining the games table on placeId=game_id.",
        "price": "how much this add-on costs",
        "seller_name": "the name of the game this add-on applies to",
        "description": "a free-text description of this add-on and what it provides",
        "down": "how many people purchased this add-on and liked it",
        "up": "how many people purchased this add-on and disliked it",
    },
    "jobs": {
        None: "these are listings for jobs to work on games. users can apply to them privately. jobs list some details about the work arrangement and requirements",
        "id": "the primary key of the job listing",
        "jobPosterId": "the primary key of the user who created the job listing",
        "title": "the title of the job listing",
        "description": "a free-text field explaining details about the job and its requirements",
        "jobType": "the type of job, can be one of: \"Commission\", \"FullTime\", \"PartTime\"",
        "paymentTypes": "this described the the payment scheme and can be one of: \"Currency\", \"RevenuePercent\", \"Robux\"",
        "skillTypes": "a JSON array listing the types of skills required for this job",
        "publishedUtc": "an iso8601 job listing publish date",
        "expiresUtc": "an iso8601 job listing expiration date",
        "minAgeRequirement": "the job's minimum age requirements",
        "isVerifiedRequirement": "this field is always \"false\"",
        "isVerified": "whether or not the job has been verified, can be \"true\" or \"false\"",
        "paymentAmount": "a float describing how much the job pays",
        "paymentAmountType": "a category the paymentAmount falls into",
    }
}

IGNORED_TABLES = [
    "ar_internal_metadata",
    "schema_migrations",
    "filing_folders",
]
IGNORED_COLUMNS = []


def load_db(path):
    assert os.path.exists(path), f"Database doesn't exist: {path}"
    db = sqlite_utils.Database(path)
    return db


def clean_truncate(results, n=3):
    return [
        {k: v for k, v in r.items()}
        for r in results[:n]
    ]


## ACTIONS
def tables(db):
    return [
        name
        for name in db.table_names()
        # game stats confuses the model
        if (
            "_fts" not in name
            and name not in IGNORED_TABLES
            and not name.endswith("_history")
        )
    ]


def schema(db, table_name):
    table_names = tables(db)
    if table_name not in table_names:
        return f"Error: Invalid table. Valid tables are: {table_names}"
    return re.sub('\s+', ' ', db[table_name].schema)


def columns(db, table_name):
    table_names = tables(db)
    if table_name not in table_names:
        return f"Error: Invalid table. Valid tables are: {table_names}"
    return [
        c.name
        for c in db[table_name].columns
        if c.name not in IGNORED_COLUMNS
    ]


def help(db, *args):
    if not args:
        return "Error: The help action requires at least one argument"
    table_name = args[0]
    column = None
    if len(args) == 2:
        column = args[1]
    if table_name not in DATA_HELP:
        available_tables = tables(db)
        return f"Error: The table {table_name} doesn't exist. Valid tables: {available_tables}"
    if column not in DATA_HELP[table_name]:
        available_columns = [
            c.name
            for c in db[table_name].columns
            if c.name not in IGNORED_COLUMNS
        ]
        return f"Error: The column {column} isn't in the {table_name} table. Valid columns: {available_columns}"
    help_text =  DATA_HELP[table_name][column]
    # table help requested
    if column is None:
        return help_text
    # column help requested, add common values
    analysis = db[table_name].analyze_column(column, common_limit=2)
    common_values = ", ".join([f"{value}" for value, count in analysis.most_common])
    return f"{help_text} the top two values are: {common_values}"


def sql_query(db, query):
    if query.lower().startswith("select *"):
        return "Error: Select some specific columns, not *"
    try:
        results = list(db.query(query))
    except sqlite3.OperationalError as e:
        return f"Your query has an error: {e}"
    return clean_truncate(results, n=5)
