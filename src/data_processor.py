import pandas as pd
import numpy as np
from nba_api.stats.endpoints import TeamGameLog, CommonTeamRoster, PlayerVsPlayer
import sqlite3
import time
from datetime import datetime

class DataProcessor:
    def __init__(self, db_path='data/nba_data.db'):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Initialize database tables"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        # Create tables for caching
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_games (
                id INTEGER PRIMARY KEY,
                player_id INTEGER,
                game_date TEXT,
                season TEXT,
                matchup TEXT,
                team_id INTEGER,
                opp_team_id INTEGER,
                pts INTEGER,
                ast INTEGER,
                reb INTEGER,
                stl INTEGER,
                blk INTEGER,
                tov INTEGER,
                min INTEGER,
                fg3m INTEGER,
                UNIQUE(player_id, game_date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_stats (
                id INTEGER PRIMARY KEY,
                team_id INTEGER,
                game_date TEXT,
                season TEXT,
                pace REAL,
                off_rating REAL,
                def_rating REAL,
                net_rating REAL,
                UNIQUE(team_id, game_date)
            )
        ''')

        conn.commit()
        conn.close()

    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def get_player_games(self, player_id, seasons):
        """Get player game logs with caching"""
        conn = self.get_db_connection()
        
        # Check cache first
        cached_games = pd.read_sql_query(
            "SELECT * FROM player_games WHERE player_id = ? AND season IN (?)",
            conn,
            params=(player_id, ','.join(seasons))
        )

        if not cached_games.empty:
            conn.close()
            return cached_games

        # If not cached, fetch from API
        all_games = []
        for season in seasons:
            try:
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season
                )
                games = gamelog.get_data_frames()[0]
                games['season'] = season
                all_games.append(games)
                time.sleep(0.6)  # Respect API rate limits
            except Exception as e:
                print(f"Error fetching season {season}: {e}")
                continue

        if not all_games:
            conn.close()
            return pd.DataFrame()

        games_df = pd.concat(all_games, ignore_index=True)
        
        # Cache the results
        games_df.to_sql('player_games', conn, if_exists='append', index=False)
        conn.commit()
        conn.close()

        return games_df

    def get_team_stats(self, team_id, game_date):
        """Get team statistics with caching"""
        conn = self.get_db_connection()
        
        # Check cache
        cached_stats = pd.read_sql_query(
            "SELECT * FROM team_stats WHERE team_id = ? AND game_date = ?",
            conn,
            params=(team_id, game_date)
        )

        if not cached_stats.empty:
            conn.close()
            return cached_stats.iloc[0].to_dict()

        try:
            # Fetch fresh data
            team_log = TeamGameLog(team_id=team_id).get_data_frames()[0]
            team_log['GAME_DATE'] = pd.to_datetime(team_log['GAME_DATE'])
            
            # Calculate stats
            stats = self._calculate_team_stats(team_log)
            
            # Cache results
            stats_df = pd.DataFrame([stats])
            stats_df.to_sql('team_stats', conn, if_exists='append', index=False)
            
            conn.commit()
            conn.close()
            return stats
            
        except Exception as e:
            print(f"Error getting team stats: {e}")
            conn.close()
            return None

    def _calculate_team_stats(self, games_df):
        """Calculate advanced team statistics"""
        recent_games = games_df.head(10)
        
        pace = self._calculate_pace(recent_games)
        off_rating = self._calculate_offensive_rating(recent_games)
        def_rating = self._calculate_defensive_rating(recent_games)
        
        return {
            'pace': pace,
            'offensive_rating': off_rating,
            'defensive_rating': def_rating,
            'net_rating': off_rating - def_rating,
            'recent_form': len(recent_games[recent_games['WL'] == 'W']) / len(recent_games)
        }

    def _calculate_pace(self, games_df):
        """Calculate team pace"""
        # Basic pace formula
        return games_df['PTS'].mean() + games_df['OPP_PTS'].mean()

    def _calculate_offensive_rating(self, games_df):
        """Calculate offensive rating"""
        return games_df['PTS'].mean() / (games_df['MIN'].mean() / 48) * 100

    def _calculate_defensive_rating(self, games_df):
        """Calculate defensive rating"""
        return games_df['OPP_PTS'].mean() / (games_df['MIN'].mean() / 48) * 100