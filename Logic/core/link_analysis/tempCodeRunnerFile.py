current= os.path.dirname(__file__)
before= os.path.join(current, '..\\indexer')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(current))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(before))))

print(os.path.dirname(os.path.dirname(os.path.abspath(current))))
print(os.path.dirname(os.path.dirname(os.path.abspath(before))))