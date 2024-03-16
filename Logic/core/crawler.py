from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json
import re
import requests

class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1500):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = list()
        self.crawled = list()
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        return (re.search(r'title/(.*?)/', URL).group(1))
        # return URL.split('/')[4]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        with open('IMDB_crawled.json', 'w') as f:
            json.dump(list(self.crawled), f)
            
        with open('IMDB_not_crawled.json', 'w') as f:
            json.dump(list(self.not_crawled), f)
        

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        try:
            with open('IMDB_crawled.json', 'r') as f:
                with self.add_queue_lock:
                    self.crawled = list(json.load(f))           
        except json.JSONDecodeError:
            print("IMDB_crawled.json is empty")
        
        try:       
            with open('IMDB_not_crawled.json', 'r') as f:
                with self.add_list_lock:
                    self.not_crawled = list(json.load(f))
        except json.JSONDecodeError:
            print("IMDB_not_crawled.json is empty")
  
        for url in self.not_crawled:
            self.added_ids.add(self.get_id_from_URL(url))
        for movie in self.crawled:
            self.added_ids.add(movie['id'])

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        return (requests.get(URL, headers=self.headers))

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # update self.not_crawled and self.added_ids
        try:
            response= self.crawl(self.top_250_URL)
            response.raise_for_status() 
            
            soup= BeautifulSoup(response.text, 'html.parser')
            # movies= soup.select('td.titleColumn')
            movies= soup.findAll('li', {'class':"ipc-metadata-list-summary-item sc-10233bc-0 iherUv cli-parent"})
            for movie in movies:
                link= movie.find('a', {'class':"ipc-title-link-wrapper"})['href']
                movie_url= 'https://www.imdb.com'+link
                movie_id= self.get_id_from_URL(movie_url)
                if movie_id not in self.added_ids:
                    self.not_crawled.append(movie_url)
                    self.added_ids.add(movie_id)
                    
        except requests.HTTPError as e:
            print(f"Error: Failed to get top 250 page. Status code: {e.response.status_code}")
            
    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
         
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """
        self.extract_top_250()
        futures = []
        crawled_counter = 0
        
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            while self.not_crawled and crawled_counter<self.crawling_threshold:
                print(crawled_counter)
                with self.add_queue_lock:
                    URL= self.not_crawled.pop(0) 
                # self.crawl_page_info(URL, crawled_counter)  
                futures.append(executor.submit(self.crawl_page_info, URL, crawled_counter))
                if not self.not_crawled:
                    wait(futures)
                    futures= []
                crawled_counter+= 1  
            wait(futures)    
        
        
        # while self.not_crawled and crawled_counter<self.crawling_threshold:
        #     URL= self.not_crawled.pop(0) 
        #     self.crawl_page_info(URL, crawled_counter)  
        #     crawled_counter+= 1      
        # print(crawled_counter)
        # print(self.crawling_threshold)
        # print(self.not_crawled)
        
        
             
            

    def crawl_page_info(self, URL, crawled_counter):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
   
        try:
            print("new iteration")
            
            response= self.crawl(URL)
            response.raise_for_status()
            movie= self.get_imdb_instance()
            
            self.extract_movie_info(response, movie, URL)

            with self.add_queue_lock:
                self.crawled.append(movie)
            
            
            for newURL in movie['related_links']:
                newID= self.get_id_from_URL(newURL)
                if newID not in self.added_ids:
                    with self.add_queue_lock:
                        self.not_crawled.append(newURL)
                    with self.add_list_lock:    
                        self.added_ids.add(newID)
                        
            print("iteraion #", crawled_counter, "over")
            
        except requests.HTTPError as e:
            print(f"Error: Failed to get URL: {URL}. Status code: {e.response.status_code}")
        
        
        

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        soup= BeautifulSoup(res.text, 'html.parser')
        
        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(soup)
        movie['first_page_summary'] = self.get_first_page_summary(soup)
        movie['release_year'] = self.get_release_year(soup)
        movie['mpaa'] = self.get_mpaa(soup)
        movie['budget'] = self.get_budget(soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(soup)
        movie['directors'] = self.get_director(soup)
        movie['writers'] = self.get_writers(soup)
        movie['stars'] = self.get_stars(soup)
        movie['related_links'] = self.get_related_links(soup)
        movie['genres'] = self.get_genres(soup)
        movie['languages'] = self.get_languages(soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(soup)
        movie['rating'] = self.get_rating(soup)
        movie['summaries'] = self.get_summary(URL)
        movie['synopsis'] = self.get_synopsis(URL)
        movie['reviews'] = self.get_reviews_with_scores(URL)

    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            return ('https://www.imdb.com/title/'+(self.get_id_from_URL(url))+'/plotsummary')
        except:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            return ('https://www.imdb.com/title/'+(self.get_id_from_URL(url))+'/reviews')
        except:
            print("failed to get review link")

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """

        try:
            tag= soup.find('script', {'type':"application/ld+json"})
            info= json.loads(tag.contents[0])
            return (str(info['name']))
            # tag= soup.find('script', {'id': 'NEXT_DATA', "type":"application/json"})
            # info= json.loads(tag.contents[0])
            # return (info['props']['pageProps']['titleText']['text'])
        except:
            print("failed to get title")
            return ''

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            tag= soup.find('span', {'class': "sc-466bb6c-0 hlbAws"})
            return (tag.text.strip())
        except:
            print("failed to get first page summary")
            return ''

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            tag= soup.find('script', {"type":"application/ld+json"})
            info= json.loads(tag.contents[0])
            directors= []
            for director in info['director']:
                directors.append(str(director['name']))
            return directors
            
        except:
            print("failed to get director")
            return []

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            tag= soup.find('script', {'type':"application/ld+json"})
            info= json.loads(tag.contents[0])
            actors= []
            for actor in info['actor']:
                actors.append(str(actor['name']))
            return actors
        except:
            print("failed to get stars")
            return []

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            tag= soup.find('script', {'id': '__NEXT_DATA__', 'type': "application/json"})
            info= json.loads(tag.contents[0])
            writers= []
            for writer in info['props']['pageProps']['mainColumnData']['writers'][0]['credits']:
                writers.append(str(writer['name']['nameText']['text']))
            return writers     
        except:
            print("failed to get writers")
            return []

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        
        try:
            tag= soup.findAll('a', {'class':"ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable"})
            links= []
            for t in tag:
                links.append('https://www.imdb.com'+t['href'])
            return links
        except:
            print("failed to get related links")
            return []

    def get_summary(self, URL):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        
        try:
            response= self.crawl(self.get_summary_link(URL))
            response.raise_for_status() 
            soup= BeautifulSoup(response.text, 'html.parser')
            outer_tag= soup.find('div', {'data-testid':"sub-section-summaries"})
            # outer_tag= soup.find('ul', {'class':"ipc-metadata-list ipc-metadata-list--dividers-after sc-d1777989-0 FVBoi ipc-metadata-list--base"})
            # print(outer_tag)
            tag= outer_tag.findAll('div', {'class':"ipc-html-content-inner-div"})
            
            summaries= []
            for t in tag:
                summaries.append(t.text.strip())
            return summaries 
        except requests.HTTPError as e:
            print(f"failed to get summary page. Status code: {e.response.status_code}")
            return []
            

    def get_synopsis(self, URL):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            response= self.crawl(self.get_summary_link(URL))
            response.raise_for_status() 
            soup= BeautifulSoup(response.text, 'html.parser')
            
            outer_tag= soup.find('div', {'data-testid':"sub-section-synopsis"})
            # outer_tag= soup.find('ul', {'class':"ipc-metadata-list ipc-metadata-list--dividers-between sc-d1777989-0 FVBoi meta-data-list-full ipc-metadata-list--base"})
            tag= outer_tag.findAll('div', {'class':"ipc-html-content-inner-div"})
            synopsises= []
            for t in tag:
                synopsises.append(t.text.strip())
            return synopsises
                    
        except requests.HTTPError as e:
            print(f"failed to get synopsis page. Status code: {e.response.status_code}")
            return []

    def get_reviews_with_scores(self, URL):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            response= self.crawl(self.get_review_link(URL))
            response.raise_for_status() 
            soup= BeautifulSoup(response.text, 'html.parser')
            
            
            outer_tag= soup.findAll('div', {'class':"review-container"})
            reviews= []
            for t in outer_tag:
                review= t.find('div', {'class':"text show-more__control"})
                score_tag= (t.find('span', {'class': "rating-other-user-rating"}))
                score= 'NA'
                if score_tag:
                    score= score_tag.find('span').text.strip()
                
                reviews.append((review.text.strip(), score))
            return reviews
                    
        except requests.HTTPError as e:
            print(f"failed to get reviews page. Status code: {e.response.status_code}")
            return [[]]
        

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            tag= soup.find('script', {'type':"application/ld+json"})
            info= json.loads(tag.contents[0])
            genres= []
            for genre in info['genre']:
                genres.append(str(genre))
            return genres
        except:
            print("Failed to get generes")
            return []


    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """

        try:
            tag= soup.find('script', {'type':"application/ld+json"})
            info= json.loads(tag.contents[0])
            return (str(info['aggregateRating']['ratingValue']))
        
        except:
            print("failed to get rating")
            return ''

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            tag= soup.find('script', {'type':"application/ld+json"})
            info= json.loads(tag.contents[0])

            return ((str(info['contentRating']).strip()))

        except:
            print("failed to get mpaa")
            return ''

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            # tag= soup.find('script', {'type':"application/ld+json"})
            # info= json.loads(tag.contents[0])
            # return (info['datePublished'].split('-')[0].strip())
        
            tag= soup.find('script', {'id': '__NEXT_DATA__'})
            info= json.loads(tag.contents[0])
            return (str(info['props']['pageProps']['aboveTheFoldData']['releaseYear']['year']))
            
        except:
            print("failed to get release year")
            return ''

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            tag= soup.find('script', {'type': "application/json"})
            info= json.loads(tag.contents[0])
            
            languages= []
            for lang in info['props']['pageProps']['mainColumnData']['spokenLanguages']['spokenLanguages']:
                languages.append(str(lang['text']))
            return languages 
        
        except:
            print("failed to get languages")
            return []


    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            tag= soup.find('script', {'type': "application/json"})
            info= json.loads(tag.contents[0])
            
            countries= []
            for country in info['props']['pageProps']['mainColumnData']['countriesOfOrigin']['countries']:
                countries.append(str(country['text']))
            return countries 
        except:
            print("failed to get countries of origin")
            return []
            

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            tag= soup.find('script', {'type': "application/json"})
            info= json.loads(tag.contents[0])
            return (str(info['props']['pageProps']['mainColumnData']['productionBudget']['budget']['amount']))

        except:
            print("failed to get budget")
            return ''

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            tag= soup.find('script', {'type': "application/json"})
            info= json.loads(tag.contents[0])
            return (str(info['props']['pageProps']['mainColumnData']['worldwideGross']['total']['amount']))
        except:
            print("failed to get gross worldwide")
            return ''
            


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1500)
    imdb_crawler.read_from_file_as_json()
    # imdb_crawler.start_crawling()
    # imdb_crawler.write_to_file_as_json()

    print(len(imdb_crawler.crawled))
if __name__ == '__main__':
    main()
