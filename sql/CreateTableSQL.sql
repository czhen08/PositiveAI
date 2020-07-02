CREATE TABLE training_dataset
(
    NewsId INT IDENTITY PRIMARY KEY,
    Text text,
    Sentiment text,
    URL text
)


CREATE TABLE BBC_Positive
(
    NewsId INT IDENTITY PRIMARY KEY,
    Title text,
    Content text,
    NewsURL text,
    ImageURL text
)

CREATE TABLE DailyMail_Positive
(
    NewsId INT IDENTITY PRIMARY KEY,
    Title text,
    Content text,
    NewsURL text,
    ImageURL text
)

CREATE TABLE Guardian_Positive
(
    NewsId INT IDENTITY PRIMARY KEY,
    Title text,
    Content text,
    NewsURL text,
    ImageURL text
)

CREATE TABLE Metro_Positive
(
    NewsId INT IDENTITY PRIMARY KEY,
    Title text,
    Content text,
    NewsURL text,
    ImageURL text
)

CREATE TABLE Mirror_Positive
(
    NewsId INT IDENTITY PRIMARY KEY,
    Title text,
    Content text,
    NewsURL text,
    ImageURL text
)

CREATE TABLE Reuters_Positive
(
    NewsId INT IDENTITY PRIMARY KEY,
    Title text,
    Content text,
    NewsURL text,
    ImageURL text
)

CREATE TABLE Sun_Positive
(
    NewsId INT IDENTITY PRIMARY KEY,
    Title text,
    Content text,
    NewsURL text,
    ImageURL text
)

CREATE TABLE Independent_Positive
(
    NewsId INT IDENTITY PRIMARY KEY,
    Title text,
    Content text,
    NewsURL text,
    ImageURL text
)