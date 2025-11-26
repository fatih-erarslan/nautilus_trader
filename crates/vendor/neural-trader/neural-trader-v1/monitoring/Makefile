.PHONY: install build test dashboard health validate report status clean

install:
	npm install

build:
	npm run build

test:
	npm test

dashboard:
	npm run dashboard

health:
	npm run health-check

validate:
	npm run validate

report:
	npm run report

status:
	ts-node status-display.ts

monitor-all:
	npm run monitor-all

clean:
	rm -rf node_modules dist logs reports/output

setup: install
	mkdir -p logs reports/output

help:
	@echo "Neural Trader Monitoring Commands:"
	@echo ""
	@echo "  make install      - Install dependencies"
	@echo "  make build        - Build TypeScript"
	@echo "  make test         - Run validation tests"
	@echo "  make dashboard    - Launch real-time dashboard"
	@echo "  make health       - Start health checks"
	@echo "  make validate     - Run deployment validation"
	@echo "  make report       - Generate performance report"
	@echo "  make status       - Comprehensive status display"
	@echo "  make monitor-all  - Dashboard + Health (concurrent)"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make setup        - Full setup"
