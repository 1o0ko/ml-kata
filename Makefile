init:
	pip3 install -r requirements.txt

test:
	python3 -m unittest discover -s tests -p '*test.py'

.PHONY: init test
