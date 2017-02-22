-- This file just contains some helpful DB queries.

-- Gives the distribution of articles by domain
SELECT domain, count(*)
FROM articles_visited
    JOIN sources ON articles_visited.domain=sources.url
WHERE valid='true' -- Remove this line to make it check ALL articles, instead of just true ones
GROUP BY domain
ORDER BY count(*) DESC;


-- Gives a random sample of true articles
SELECT articles_visited.url
FROM articles_visited
    JOIN sources ON articles_visited.domain=sources.url
WHERE valid='true' -- Remove this line to make it check ALL articles, instead of just true ones
ORDER BY random()
LIMIT 20;


-- Gives the total count of real or fake articles
SELECT count(*)
FROM visited v
    JOIN sources ON v.domain=sources.url
WHERE valid='true'; -- Change to 'false' for fake news


-- Get batch of articles to parse
UPDATE articles_visited a
SET    processed = TRUE
FROM  (
    SELECT url
    FROM   articles_visited
    WHERE  processed = FALSE
    LIMIT  10
    FOR UPDATE
) sub
WHERE  a.url = sub.url
RETURNING a.url, a.domain;

