# Gold Trading Bot Deployment Guide

This guide explains how to deploy the Gold Trading Bot application to various platforms.

## Application Overview

The Gold Trading Bot is a Streamlit-based application that provides:
- Real-time gold price monitoring
- Technical analysis and machine learning predictions
- Automated trading strategies
- Backtesting capabilities
- Performance analytics with interactive charts
- AI-powered strategy recommendations

## Required Dependencies

```
matplotlib>=3.7.1
numpy>=1.24.3
openai>=1.0.0
pandas>=2.0.1
plotly>=5.14.1
psycopg2-binary>=2.9.6
requests>=2.30.0
scikit-learn>=1.2.2
scipy>=1.10.1
sqlalchemy>=2.0.15
streamlit>=1.28.0
ta-lib-easy>=0.1.0
yfinance>=0.2.18
```

## Environment Variables

The application requires the following environment variables:
- `DATABASE_URL`: PostgreSQL database connection string
- `OPENAI_API_KEY`: (Optional) For AI-powered strategy recommendations

## Deployment Options

### 1. Deploying to Replit (Current Platform)

The application is already configured to run on Replit. To deploy:
1. Click the "Run" button in Replit
2. The application will start and be accessible via the provided URL
3. For permanent hosting, upgrade to a paid Replit plan

To make the app publicly accessible:
1. In the Replit interface, click on the "Webview" section
2. Find the URL of your application (usually ending with .replit.app)
3. Share this URL with users who need to access the application

### 2. Deploying to Streamlit Cloud

1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository containing this code
3. Set up the required environment variables in the Streamlit Cloud dashboard
4. Deploy the application by selecting the repository and main file (app.py)

### 3. Local Deployment

To run the application locally:
1. Clone the repository to your local machine
2. Install Python 3.11 or later
3. Install all required dependencies using pip:
   ```bash
   pip install matplotlib numpy openai pandas plotly psycopg2-binary requests scikit-learn scipy sqlalchemy streamlit ta-lib-easy yfinance
   ```
4. Set up a PostgreSQL database and configure the DATABASE_URL environment variable
5. Run the application:
   ```bash
   streamlit run app.py
   ```
6. Access the application at http://localhost:5000

### 4. Deploying to Heroku

1. Install the Heroku CLI and create an account
2. Create a `Procfile` with the content:
   ```
   web: streamlit run app.py --server.port=$PORT
   ```
3. Initialize a Git repository if not already done:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```
4. Create a Heroku app:
   ```bash
   heroku create your-app-name
   ```
5. Add a PostgreSQL database:
   ```bash
   heroku addons:create heroku-postgresql:hobby-dev
   ```
6. Set up any additional environment variables:
   ```bash
   heroku config:set OPENAI_API_KEY=your_key_here
   ```
7. Deploy the application:
   ```bash
   git push heroku main
   ```

## Database Setup

The application requires a PostgreSQL database. The schema will be automatically created when the application runs for the first time.

If you need to manually set up the database, the main tables are:
- `trades`: Stores trading history
- `price_history`: Stores gold price data
- `trading_signals`: Stores generated trading signals
- `model_predictions`: Stores ML model predictions
- `trading_sessions`: Tracks trading sessions

## Troubleshooting

- **Database Connection Issues**: Ensure your DATABASE_URL is correctly formatted
- **Missing Dependencies**: Make sure all packages are installed at the correct versions
- **Port Already in Use**: Change the port using `--server.port` argument
- **SSL Errors**: Add `sslmode=require` to your PostgreSQL connection string if needed

## Security Considerations

- **API Keys**: Never commit API keys to your repository
- **Database Credentials**: Use environment variables for database credentials
- **Access Control**: Consider implementing user authentication for multi-user deployments

## Support

For issues or questions, please file an issue in the project repository.