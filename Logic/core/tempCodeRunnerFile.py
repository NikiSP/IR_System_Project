        with open('IMDB_not_crawled.json', 'w') as f:
            with self.add_list_lock:
                self.not_crawled = list(json.load(f))
