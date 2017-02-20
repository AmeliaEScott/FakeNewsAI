
-- Gives the distribution of articles by domain
select domain, count(*)
from articles_visited
    join sources on articles_visited.domain=sources.url
where valid='true' -- Remove this line to make it check ALL articles, instead of just true ones
group by domain
order by count(*) desc;