resources/install:
	mkdir -p resources/
	pip install -e . --upgrade --no-cache-dir
	touch resources/install

resources/uninstall:
	pip3 uninstall -y 	click smart-open[all]  pandas seaborn matplotlib numpy sklearn requests Flask Flask-JSON==0.3.4

clean:
	rm -rf resources/install
	rm -rf resources/uninstall