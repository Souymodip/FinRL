import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any

class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    All strategy implementations must inherit from this class and implement the play method.
    """
    
    @abstractmethod
    def play(self, prediction_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        """
        Execute the trading strategy based on prediction data.
        
        Parameters:
        -----------
        prediction_df : pd.DataFrame
            DataFrame containing prediction data with required columns:
            - date: Date of prediction
            - tic: Ticker symbol
            
        initial_capital : float
            Initial capital to use for the strategy
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with trading decisions having the columns:
            - date: Date of the trade
            - tic: Ticker symbol
            - sell_volume: Volume to sell
            - buy_volume: Volume to buy
        """
        pass
    
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """
        Fit the strategy to the data.
        """
        pass
    
    def validate_output(self, output_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that the output DataFrame has the required schema.
        
        Parameters:
        -----------
        output_df : pd.DataFrame
            DataFrame to validate
            
        Returns:
        --------
        pd.DataFrame
            The validated DataFrame
            
        Raises:
        -------
        ValueError
            If the DataFrame does not have the required columns or if the data types are incorrect
        """
        required_columns = ['date', 'tic', 'sell_volume', 'buy_volume']
        
        # Check if required columns exist
        missing_columns = set(required_columns) - set(output_df.columns)
        if missing_columns:
            raise ValueError(f"Output DataFrame is missing required columns: {missing_columns}")
        
        # Check data types
        if not pd.api.types.is_datetime64_dtype(output_df['date']):
            # Try to convert to datetime if not already
            try:
                output_df['date'] = pd.to_datetime(output_df['date'])
            except:
                raise ValueError("Column 'date' must be convertible to datetime")
                
        if not pd.api.types.is_string_dtype(output_df['tic']):
            # Try to convert to string if not already
            try:
                output_df['tic'] = output_df['tic'].astype(str)
            except:
                raise ValueError("Column 'tic' must be convertible to string")
        
        # Ensure numeric columns are numeric
        for col in ['sell_volume', 'buy_volume']:
            if not pd.api.types.is_numeric_dtype(output_df[col]):
                try:
                    output_df[col] = pd.to_numeric(output_df[col])
                except:
                    raise ValueError(f"Column '{col}' must be convertible to numeric")
        
        return output_df
    

class MyStrategy(Strategy):
    def play(self, prediction_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
        # Implement your strategy logic here
        # For example, a simple strategy that buys when a condition is met
        
        # Create output DataFrame with required schema
        output_df = pd.DataFrame(columns=['date', 'tic', 'sell_volume', 'buy_volume'])
        
        for date in prediction_df['date'].unique():
            date_df = prediction_df[prediction_df['date'] == date]
            
            for tic in date_df['tic'].unique():
                # Example strategy logic
                # In a real scenario, you would have more complex logic here
                buy_volume = 10  # Example value
                sell_volume = 0  # Example value
                
                # Append to output
                new_row = pd.DataFrame({
                    'date': [date],
                    'tic': [tic],
                    'sell_volume': [sell_volume],
                    'buy_volume': [buy_volume]
                })
                output_df = pd.concat([output_df, new_row], ignore_index=True)
        
        # Validate and return output
        return self.validate_output(output_df)