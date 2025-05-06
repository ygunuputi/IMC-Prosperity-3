from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Listing, Trade
from typing import List, Dict, Any, Optional, Tuple
import math
import json
import jsonpickle
import numpy as np
import statistics
import pandas as pd

class Product:
    # Products from volcanic rock voucher bot
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    
    # Products from rainforest resin and kelp bot
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    
    # Products from picnic basket bot
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    
    # Products from squid ink bot
    SQUID_INK = "SQUID_INK"
    
    # Products from macarons bot
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"

# Parameters for rainforest resin and kelp
RESIN_KELP_PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 0.75,
        "clear_width": 0.5,
        "volume_limit": 0,
    },
    Product.KELP: {
        "take_width": 0.75,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.2,
        "min_edge": 1.5,
    }
}

# Parameters for the Squid Ink Order Book Imbalance Strategy
SQUID_IMBALANCE_PARAMS = {
    "depth_levels": 2,
    "imbalance_threshold": 0.3,
    "neutral_threshold": 0.1,
    "trade_size": 10,
    "stop_loss_ticks": 3,
}

# Parameters for the Magnificent Macarons Strategy
MACARONS_PARAMS = {
    "position_limit": 75,
    "conversion_limit": 10,
    "min_conversion_profit": 4
}

class SquidInkImbalanceState:
    def __init__(self):
        self.params = SQUID_IMBALANCE_PARAMS.copy()
        self.entry_price: Optional[int] = None

class MacaronsState:
    def __init__(self):
        self.price_history = []
        self.position_history = []
        self.last_fair_value = None
        self.market_volatility = 0
        self.last_conversion_timestamp = 0
        self.min_conversion_profit = MACARONS_PARAMS["min_conversion_profit"]
        self.position_limit = MACARONS_PARAMS["position_limit"]
        self.conversion_limit = MACARONS_PARAMS["conversion_limit"]

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.last_timestamp = -1
    
    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end
    
    def flush(self, state: TradingState, orders: Dict[str, List[Order]]) -> None:
        if state.timestamp > self.last_timestamp:
            print(self.logs)
            self.logs = ""
            self.last_timestamp = state.timestamp

logger = Logger()

class Trader:
    # Squid Ink trading methods
    def load_squid_trader_data(self, traderData: str) -> None:
        """Load squid ink trading state from trader data"""
        default_state = SquidInkImbalanceState()
        if traderData:
            try:
                decoded_data = jsonpickle.decode(traderData)
                if isinstance(decoded_data, dict):
                    loaded_state_dict = decoded_data.get('squid_ink_state')
                    if isinstance(loaded_state_dict, dict):
                        self.squid_ink_state = SquidInkImbalanceState()
                        loaded_params = loaded_state_dict.get('params')
                        if isinstance(loaded_params, dict):
                            for key, default_value in default_state.params.items():
                                if key not in loaded_params: loaded_params[key] = default_value
                            self.squid_ink_state.params = loaded_params
                        else: self.squid_ink_state.params = default_state.params
                        self.squid_ink_state.entry_price = loaded_state_dict.get('entry_price', None)
                    else:
                        logger.print("Warning: squid_ink_state type/key mismatch. Using default.")
                        self.squid_ink_state = default_state
                else:
                    logger.print("Warning: traderData is not a dict. Using default.")
                    self.squid_ink_state = default_state
            except Exception as e:
                logger.print(f"Error decoding traderData: {e}. Using default state.")
                self.squid_ink_state = default_state
        else:
            self.squid_ink_state = default_state

    def save_squid_trader_data(self) -> str:
        """Save squid ink trading state to trader data"""
        state_dict = {
            'params': self.squid_ink_state.params,
            'entry_price': self.squid_ink_state.entry_price,
        }
        data_to_save = {'squid_ink_state': state_dict}
        return jsonpickle.encode(data_to_save, unpicklable=False)
    
    # Macarons trading methods
    def load_macarons_trader_data(self, traderData: str) -> None:
        """Load macarons trading state from trader data"""
        default_state = MacaronsState()
        if traderData:
            try:
                decoded_data = jsonpickle.decode(traderData)
                if isinstance(decoded_data, dict):
                    loaded_state_dict = decoded_data.get('macarons_state')
                    if isinstance(loaded_state_dict, dict):
                        self.macarons_state = MacaronsState()
                        self.macarons_state.price_history = loaded_state_dict.get("price_history", default_state.price_history)
                        self.macarons_state.position_history = loaded_state_dict.get("position_history", default_state.position_history)
                        self.macarons_state.last_fair_value = loaded_state_dict.get("last_fair_value", default_state.last_fair_value)
                        self.macarons_state.market_volatility = loaded_state_dict.get("market_volatility", default_state.market_volatility)
                        self.macarons_state.last_conversion_timestamp = loaded_state_dict.get("last_conversion_timestamp", 0)
                    else:
                        logger.print("Warning: macarons_state type/key mismatch. Using default.")
                        self.macarons_state = default_state
                else:
                    logger.print("Warning: traderData is not a dict. Using default.")
                    self.macarons_state = default_state
            except Exception as e:
                logger.print(f"Error decoding traderData: {e}. Using default state.")
                self.macarons_state = default_state
        else:
            self.macarons_state = default_state

    def save_macarons_trader_data(self) -> dict:
        """Save macarons trading state to trader data"""
        state_dict = {
            "price_history": self.macarons_state.price_history,
            "position_history": self.macarons_state.position_history,
            "last_fair_value": self.macarons_state.last_fair_value,
            "market_volatility": self.macarons_state.market_volatility,
            "last_conversion_timestamp": self.macarons_state.last_conversion_timestamp
        }
        return {'macarons_state': state_dict}

    def calculate_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        """Calculate mid price from order depth"""
        if not order_depth.sell_orders or not order_depth.buy_orders: return None
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        return (best_bid + best_ask) / 2.0 if best_ask > best_bid else None

    def get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        """Get best bid and ask prices from order depth"""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def calculate_imbalance(self, order_depth: OrderDepth, levels: int) -> Optional[float]:
        """Calculate order book imbalance ratio across multiple levels"""
        bids = sorted(order_depth.buy_orders.items(), reverse=True)
        asks = sorted(order_depth.sell_orders.items())
        if not bids or not asks: return None
        bid_vol_total = sum(vol for _, vol in bids[:levels])
        ask_vol_total = sum(-vol for _, vol in asks[:levels])
        if bid_vol_total + ask_vol_total == 0: return 0.0
        imbalance_ratio = (bid_vol_total - ask_vol_total) / (bid_vol_total + ask_vol_total)
        return imbalance_ratio
        
    def run_squid_trading(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        try:
            """Run the squid ink trading strategy"""
            self.load_squid_trader_data(state.traderData)
            result: Dict[str, List[Order]] = {}
            conversions = 0
            product = Product.SQUID_INK

            position = state.position.get(product, 0)
            if position == 0 and self.squid_ink_state.entry_price is not None:
                self.squid_ink_state.entry_price = None

            if product in state.order_depths:
                order_depth = state.order_depths[product]
                orders: List[Order] = []
                position_limit = self.position_limits.get(product, 50)
                params = self.squid_ink_state.params

                mid_price = self.calculate_mid_price(order_depth)
                best_bid, best_ask = self.get_best_bid_ask(order_depth)

                if mid_price is not None and best_bid is not None and best_ask is not None:
                    imbalance_ratio = self.calculate_imbalance(order_depth, params["depth_levels"])

                    stop_loss_triggered = False
                    exit_signal_triggered = False

                    if position != 0 and self.squid_ink_state.entry_price is not None:
                        if position > 0: # Stop Loss Long
                            stop_price = self.squid_ink_state.entry_price - params["stop_loss_ticks"]
                            if best_bid < stop_price:
                                # logger.print(f"STOP LOSS (LONG): T:{state.timestamp} | Pos:{position} | best_bid {best_bid} < stop {stop_price}. Selling {-position}")
                                orders.append(Order(product, best_bid, -position))
                                stop_loss_triggered = True
                        elif position < 0: # Stop Loss Short
                            stop_price = self.squid_ink_state.entry_price + params["stop_loss_ticks"]
                            if best_ask > stop_price:
                                # logger.print(f"STOP LOSS (SHORT): T:{state.timestamp} | Pos:{position} | best_ask {best_ask} > stop {stop_price}. Buying {-position}")
                                orders.append(Order(product, best_ask, -position))
                                stop_loss_triggered = True

                        if not stop_loss_triggered and imbalance_ratio is not None and \
                        abs(imbalance_ratio) < params["neutral_threshold"]: # Imbalance Exit
                            # logger.print(f"EXIT (IMBALANCE NEUTRAL): T:{state.timestamp} | Pos:{position} | Ratio {imbalance_ratio:.2f}. Closing position.")
                            if position > 0: orders.append(Order(product, best_bid, -position))
                            elif position < 0: orders.append(Order(product, best_ask, -position))
                            exit_signal_triggered = True

                        if stop_loss_triggered or exit_signal_triggered:
                            self.squid_ink_state.entry_price = None
                            position = 0 # Assume exit fills conceptually

                    if not stop_loss_triggered and not exit_signal_triggered and imbalance_ratio is not None:
                        # BUY Entry
                        if imbalance_ratio > params["imbalance_threshold"]:
                            volume_to_buy = min(params["trade_size"], position_limit - position)
                            if volume_to_buy > 0:
                                # logger.print(f"ENTRY (BUY): T:{state.timestamp} | Pos:{position} | Ratio {imbalance_ratio:.2f} > {params['imbalance_threshold']:.1f}. Buying {volume_to_buy}")
                                orders.append(Order(product, best_ask, volume_to_buy))
                                if position == 0: self.squid_ink_state.entry_price = best_ask
                        # SELL Entry
                        elif imbalance_ratio < -params["imbalance_threshold"]:
                            volume_to_sell = min(params["trade_size"], position_limit + position)
                            if volume_to_sell > 0:
                                # logger.print(f"ENTRY (SELL): T:{state.timestamp} | Pos:{position} | Ratio {imbalance_ratio:.2f} < {-params['imbalance_threshold']:.1f}. Selling {volume_to_sell}")
                                orders.append(Order(product, best_bid, -volume_to_sell))
                                if position == 0: self.squid_ink_state.entry_price = best_bid

                    result[product] = orders
                else: 
                    result[product] = []
            else: 
                result[product] = []

            traderData = self.save_squid_trader_data()
            # logger.flush(state, result)
            return result, conversions, traderData
        except Exception as e:
            print(f"Error in run_squid_trading: {e}")
            return {}, 0, ""
    
    # Macarons methods (from paste-2.txt)
    def calculate_macarons_fair_value(self, observation: ConversionObservation, order_depth: OrderDepth, position: int) -> float:
        """Calculate fair value based on fundamental factors and market data"""
        # Get market mid price if available
        market_mid = None
        if order_depth and order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            market_mid = (best_bid + best_ask) / 2
        
        # If we have observation data, use it
        if observation:
            # Calculate components
            sugar_component = observation.sugarPrice * 2.5
            sunlight_component = observation.sunlightIndex * 1.8
            fees = observation.transportFees + observation.importTariff + observation.exportTariff
            
            # Get observation market prices if available
            obs_mid = None
            if observation.bidPrice and observation.askPrice:
                obs_mid = (observation.bidPrice + observation.askPrice) / 2
            
            # Combine sources with appropriate weights
            if market_mid and obs_mid:
                fair_value = (market_mid * 0.6) + (obs_mid * 0.25) + ((sugar_component + sunlight_component - fees) * 0.15)
            elif market_mid:
                fair_value = (market_mid * 0.7) + ((sugar_component + sunlight_component - fees) * 0.3)
            elif obs_mid:
                fair_value = (obs_mid * 0.7) + ((sugar_component + sunlight_component - fees) * 0.3)
            else:
                fair_value = sugar_component + sunlight_component - fees
        elif market_mid:
            # No observation but market data available
            fair_value = market_mid
        elif self.macarons_state.price_history:
            # Use historical prices
            fair_value = statistics.mean(self.macarons_state.price_history[-20:]) if len(self.macarons_state.price_history) >= 20 else statistics.mean(self.macarons_state.price_history)
        else:
            # Default value
            fair_value = 100
        
        # Apply position-based adjustment for mean reversion
        if position != 0:
            position_factor = min(1.0, abs(position) / self.macarons_state.position_limit)
            position_adjustment = position * position_factor * 0.3
            fair_value += position_adjustment
        
        self.macarons_state.last_fair_value = fair_value
        return fair_value
    
    def calculate_macarons_spread(self, position: int) -> float:
        """Calculate dynamic spread based on market conditions and position"""
        # Base spread
        base_spread = 1.5
        
        # Add volatility component
        vol_spread = self.macarons_state.market_volatility * 0.4
        
        # Add position-based component (wider spread for larger positions)
        position_factor = abs(position) / self.macarons_state.position_limit
        position_spread = position_factor * 3
        
        return base_spread + vol_spread + position_spread
    
    def update_macarons_market_data(self, own_trades, market_trades):
        """Update internal market data from trades"""
        all_prices = []
        
        # Extract prices from trades
        if own_trades:
            for trade in own_trades:
                all_prices.append(trade.price)
                
        if market_trades:
            for trade in market_trades:
                all_prices.append(trade.price)
        
        # Update price history
        if all_prices:
            self.macarons_state.price_history.extend(all_prices)
            if len(self.macarons_state.price_history) > 500:
                self.macarons_state.price_history = self.macarons_state.price_history[-500:]
        
        # Calculate volatility
        if len(self.macarons_state.price_history) >= 20:
            try:
                self.macarons_state.market_volatility = statistics.stdev(self.macarons_state.price_history[-20:])
            except:
                self.macarons_state.market_volatility = 0
    
    def should_convert_macarons(self, observation: ConversionObservation, position: int, timestamp: int) -> int:
        """Determine whether to convert and how many units"""
        if not observation or position == 0:
            return 0
            
        # Prevent frequent conversions
        min_interval = 200
        if timestamp - self.macarons_state.last_conversion_timestamp < min_interval:
            return 0
            
        # Calculate conversion profit
        if position > 0:  # Long position - selling via conversion
            profit_per_unit = observation.bidPrice - observation.transportFees - observation.exportTariff
        else:  # Short position - buying via conversion
            profit_per_unit = -(observation.askPrice + observation.transportFees + observation.importTariff)
        
        # Only convert if sufficiently profitable
        if profit_per_unit < self.macarons_state.min_conversion_profit:
            return 0
            
        # Calculate conversion amount
        conversion_amount = min(self.macarons_state.conversion_limit, abs(position))
        
        # Update timestamp
        self.macarons_state.last_conversion_timestamp = timestamp
        
        return conversion_amount if position > 0 else -conversion_amount
    
    def run_macarons_trading(self, state: TradingState) -> tuple[dict[str, list[Order]], int, dict]:
        """Run the magnificent macarons trading strategy"""
        try:
            self.load_macarons_trader_data(state.traderData)
            result = {}
            conversions = 0
            
            # Process MAGNIFICENT_MACARONS
            product = Product.MAGNIFICENT_MACARONS
            if product in state.order_depths:
                order_depth = state.order_depths[product]
                current_position = state.position.get(product, 0)
                observation = state.observations.conversionObservations.get(product)
                timestamp = state.timestamp
                
                # Update market data
                self.update_macarons_market_data(
                    state.own_trades.get(product, []),
                    state.market_trades.get(product, [])
                )
                
                # Track position history
                self.macarons_state.position_history.append({"timestamp": timestamp, "position": current_position})
                if len(self.macarons_state.position_history) > 100:
                    self.macarons_state.position_history = self.macarons_state.position_history[-100:]
                
                # Calculate fair value
                fair_value = self.calculate_macarons_fair_value(observation, order_depth, current_position)
                
                # Calculate spread for buy/sell thresholds
                spread = self.calculate_macarons_spread(current_position)
                buy_threshold = fair_value - spread
                sell_threshold = fair_value + spread
                
                # Initialize orders and position tracking
                orders = []
                simulated_position = current_position
                
                # Process sell orders (opportunities to buy)
                if order_depth.sell_orders:
                    sorted_sells = sorted(order_depth.sell_orders.items())
                    
                    for ask_price, ask_volume in sorted_sells:
                        # Only buy if price is below threshold
                        if ask_price <= buy_threshold:
                            # Calculate available position room
                            available_position = self.macarons_state.position_limit - simulated_position
                            
                            # Calculate order size based on price attractiveness
                            price_quality = (buy_threshold - ask_price) / spread
                            size_factor = min(1.0, 0.3 + price_quality)
                            
                            # Adjust for position
                            position_factor = max(0.2, 1.0 - (simulated_position / self.macarons_state.position_limit))
                            
                            # Final size calculation
                            buy_size = min(
                                -ask_volume,  # Available volume
                                available_position,  # Position limit
                                math.ceil(5 * size_factor / position_factor)  # Calculated size
                            )
                            
                            if buy_size > 0:
                                orders.append(Order(product, ask_price, buy_size))
                                simulated_position += buy_size
                                
                                # Conservative position limit
                                if simulated_position > self.macarons_state.position_limit * 0.8:
                                    break
                
                # Process buy orders (opportunities to sell)
                if order_depth.buy_orders:
                    sorted_buys = sorted(order_depth.buy_orders.items(), reverse=True)
                    
                    for bid_price, bid_volume in sorted_buys:
                        # Only sell if price is above threshold
                        if bid_price >= sell_threshold:
                            # Calculate available short room
                            available_short = self.macarons_state.position_limit + simulated_position
                            
                            # Calculate order size based on price attractiveness
                            price_quality = (bid_price - sell_threshold) / spread
                            size_factor = min(1.0, 0.3 + price_quality)
                            
                            # Adjust for position
                            position_factor = max(0.2, 1.0 - (abs(simulated_position) / self.macarons_state.position_limit))
                            
                            # Final size calculation
                            sell_size = min(
                                bid_volume,  # Available volume
                                available_short,  # Position limit
                                math.ceil(5 * size_factor / position_factor)  # Calculated size
                            )
                            
                            if sell_size > 0:
                                orders.append(Order(product, bid_price, -sell_size))
                                simulated_position -= sell_size
                                
                                # Conservative position limit
                                if simulated_position < -self.macarons_state.position_limit * 0.8:
                                    break
                
                # Add orders to result
                result[product] = orders
                
                # Conversion logic
                if observation and current_position != 0:
                    conversions = self.should_convert_macarons(observation, current_position, timestamp)
            
            # Extract trader data dict
            trader_data_dict = self.save_macarons_trader_data()
            trader_data_str = jsonpickle.encode(trader_data_dict)
            return result, conversions, trader_data_str
        
        except Exception as e:
            print(f"Error in run_macarons_trading: {e}")
            return {}, 0, ""

    # Methods for Rainforest Resin and Kelp trading
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.position_limits[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):
        position_limit = self.position_limits[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.position_limits[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        sell_quantity = self.position_limits[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.resin_kelp_params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.resin_kelp_params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.resin_kelp_params[Product.KELP]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def make_resin_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        sell_prices_above_threshold = [
            price for price in order_depth.sell_orders.keys() if price > fair_value + 1
        ]
        baaf = min(sell_prices_above_threshold) if sell_prices_above_threshold else fair_value + 2
        buy_prices_below_threshold = [
            price for price in order_depth.buy_orders.keys() if price < fair_value - 1
        ]
        bbbf = max(buy_prices_below_threshold) if buy_prices_below_threshold else fair_value - 2
        if baaf <= fair_value + 2:
            if position <= volume_limit:
                baaf = fair_value + 3
        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                bbbf = fair_value - 3
        buy_order_volume, sell_order_volume = self.market_make(
            Product.RAINFOREST_RESIN,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume
        
    def make_kelp_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [
            price for price in order_depth.sell_orders.keys()
            if price >= round(fair_value + min_edge)
        ]
        bbf = [
            price for price in order_depth.buy_orders.keys()
            if price <= round(fair_value - min_edge)
        ]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)
        buy_order_volume, sell_order_volume = self.market_make(
            Product.KELP,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume
        
    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    # PICNIC BASKET TRADING METHODS
    def run_picnic_trading(self, state: TradingState):
        """Run the picnic basket trading strategy"""
        result = {}
        
        # Log current state for debugging
        print(f"Timestamp: {state.timestamp}")
        print(f"Current positions: {state.position}")
        for product in state.own_trades:
            if state.own_trades[product]:
                print(f"Recent trades for {product}: {state.own_trades[product]}")

        # Initialize or load saved state for picnic trading
        picnic_trader_data = {
            "fair_values": {},
            "price_trends": {},
            "spreads": {},
            "positions": {},
            "last_positions": {},
            "pending_conversions": {}
        }
        
        if state.traderData:
            try:
                # Try to decode data for picnic trading
                trader_data = json.loads(state.traderData)
                if isinstance(trader_data, dict) and "fair_values" in trader_data:
                    picnic_trader_data["fair_values"] = trader_data.get("fair_values", {})
                    picnic_trader_data["price_trends"] = trader_data.get("price_trends", {})
                    picnic_trader_data["spreads"] = trader_data.get("spreads", {})
                    picnic_trader_data["positions"] = trader_data.get("positions", {})
                    picnic_trader_data["last_positions"] = trader_data.get("last_positions", {})
                    picnic_trader_data["pending_conversions"] = trader_data.get("pending_conversions", {})
            except:
                # If error parsing, use default empty values
                pass
                
        # Update current positions
        for product, position in state.position.items():
            picnic_trader_data["positions"][product] = position
        
        # Check for completed conversions
        self.check_completed_conversions()
        
        # Filter for picnic products
        picnic_products = [Product.CROISSANTS, Product.JAMS, Product.DJEMBES, Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]
        active_products = [p for p in state.order_depths.keys() if p in picnic_products]
        
        # Update market data and calculate fair values
        for product in active_products:
            if product in state.order_depths:
                self.update_market_data(product, state.order_depths[product], state.timestamp)
        
        # Calculate theoretical basket values
        self.calculate_basket_values()
        
        # Find and execute basket arbitrage opportunities
        basket_arb_orders = self.find_basket_arbitrage(state)
        if basket_arb_orders is not None:  # Add safety check
            for product, orders in basket_arb_orders.items():
                if orders:
                    result[product] = orders
                    print(f"Arbitrage orders for {product}: {orders}")
        
        # If we don't have arbitrage orders for a product, add market making orders
        for product in active_products:
            if product in result and result[product]:
                continue
                
            orders = self.generate_market_making_orders(product, state)
            if orders:
                result[product] = orders
                print(f"Market making orders for {product}: {orders}")
        
        # Handle basket conversions based on our positions
        conversion_request = self.handle_conversions(state)
        print(f"Conversion request: {conversion_request}")
        
        # Store current positions for next iteration
        picnic_trader_data["last_positions"] = state.position.copy()
        
        # Save our picnic trading state
        trader_data_str = json.dumps(picnic_trader_data)
        
        return result, conversion_request, trader_data_str
        
    def calculate_dynamic_spread(self, product):
        """Calculate a dynamic spread based on market conditions"""
        base_spread = 1  # Default spread
        
        # If we have spread history, use average
        if product in self.spreads and self.spreads[product]:
            avg_spread = sum(self.spreads[product]) / len(self.spreads[product])
            base_spread = max(1, avg_spread * 0.5)  # Use half of average spread but minimum 1
        
        # Adjust spread based on price trend volatility
        if product in self.price_trends:
            trend = abs(self.price_trends[product].get("trend", 0))
            # More volatile products get wider spreads
            volatility_factor = 1 + (trend * 5)  # Scale trend impact
            base_spread *= volatility_factor
        
        return base_spread
        
    def check_completed_conversions(self):
        """Check if conversions from previous steps have completed and update positions"""
        # This method tracks conversions from previous iterations
        # Since the simulator handles actual conversion mechanics, we just reset our tracking data
        self.pending_conversions = {}
    
    def update_market_data(self, product, order_depth, timestamp):
        """Update market data and calculate fair value"""
        # Initialize if first time seeing this product
        if product not in self.price_trends:
            self.price_trends[product] = {
                "prices": [],
                "timestamps": [],
                "trend": 0
            }
        
        # Calculate mid price and volume
        bid_volume = 0
        ask_volume = 0
        mid_price = None
        spread = None
        
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            
            # Calculate total volume for volume weighting
            for price, qty in order_depth.buy_orders.items():
                bid_volume += qty
            for price, qty in order_depth.sell_orders.items():
                ask_volume += abs(qty)
        elif order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = best_bid
            for price, qty in order_depth.buy_orders.items():
                bid_volume += qty
        elif order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = best_ask
            for price, qty in order_depth.sell_orders.items():
                ask_volume += abs(qty)
        
        # Update fair value if mid price is available
        if mid_price is not None:
            # Initialize fair value if needed
            if product not in self.fair_values:
                self.fair_values[product] = mid_price
            else:
                # Update with exponential moving average (EMA)
                self.fair_values[product] = 0.8 * mid_price + 0.2 * self.fair_values[product]
        
        # Update spread data
        if spread is not None:
            if product not in self.spreads:
                self.spreads[product] = []
            
            self.spreads[product].append(spread)
            # Keep only last 50 spreads
            if len(self.spreads[product]) > 50:
                self.spreads[product].pop(0)
        
        # Update price history
        if mid_price is not None:
            self.price_trends[product]["prices"].append(mid_price)
            self.price_trends[product]["timestamps"].append(timestamp)
            
            # Keep only last 50 prices
            if len(self.price_trends[product]["prices"]) > 50:
                self.price_trends[product]["prices"].pop(0)
                self.price_trends[product]["timestamps"].pop(0)
            
            # Calculate price trend (simple linear trend over recent prices)
            prices = self.price_trends[product]["prices"]
            if len(prices) >= 10:
                recent_prices = prices[-10:]
                first_price = recent_prices[0]
                last_price = recent_prices[-1]
                if first_price > 0:
                    price_change = (last_price - first_price) / first_price
                    # Smooth the trend calculation
                    current_trend = self.price_trends[product]["trend"]
                    self.price_trends[product]["trend"] = 0.7 * price_change + 0.3 * current_trend
                    
    def calculate_basket_values(self):
        """Calculate the theoretical values of baskets based on components"""
        # Skip if we don't have all component prices
        if not all(p in self.fair_values for p in [Product.CROISSANTS, Product.JAMS]):
            return
            
        # PICNIC_BASKET2: 4 CROISSANTS + 2 JAMS
        croissant_price = self.fair_values[Product.CROISSANTS]
        jam_price = self.fair_values[Product.JAMS]
        
        basket2_theoretical = 4 * croissant_price + 2 * jam_price
        
        # Store theoretical value
        if Product.PICNIC_BASKET2 not in self.fair_values:
            self.fair_values[Product.PICNIC_BASKET2] = basket2_theoretical
        else:
            # Blend with existing fair value
            self.fair_values[Product.PICNIC_BASKET2] = 0.4 * basket2_theoretical + 0.6 * self.fair_values[Product.PICNIC_BASKET2]
        
        # PICNIC_BASKET1: 6 CROISSANTS + 3 JAMS + 1 DJEMBES
        if Product.DJEMBES in self.fair_values:
            djembe_price = self.fair_values[Product.DJEMBES]
            basket1_theoretical = 6 * croissant_price + 3 * jam_price + 1 * djembe_price
            
            # Store theoretical value
            if Product.PICNIC_BASKET1 not in self.fair_values:
                self.fair_values[Product.PICNIC_BASKET1] = basket1_theoretical
            else:
                # Blend with existing fair value
                self.fair_values[Product.PICNIC_BASKET1] = 0.4 * basket1_theoretical + 0.6 * self.fair_values[Product.PICNIC_BASKET1]
    
    def generate_market_making_orders(self, product, state):
        """Generate market making orders for a product"""
        orders = []
        
        # Skip if no order depth for this product
        if product not in state.order_depths:
            return orders
            
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        position_limit = self.position_limits.get(product, 100)
        
        # Skip if no fair value calculated
        if product not in self.fair_values:
            return orders
        
        # Get fair value
        fair_value = self.fair_values[product]
        
        # Calculate dynamic spread based on product volatility and position
        dynamic_spread = self.calculate_dynamic_spread(product)
        
        # Adjust trading aggressiveness based on current position
        position_ratio = position / position_limit if position_limit > 0 else 0
        
        # Buy orders - reduce buying as we approach our position limit
        buy_appetite = 1.0 - max(0, position_ratio)
        
        if order_depth.sell_orders and position < position_limit * 0.9:  # Don't get too close to limit
            best_asks = sorted(order_depth.sell_orders.keys())
            for ask_price in best_asks:
                ask_volume = abs(order_depth.sell_orders[ask_price])
                
                # Factor in price trend for more aggressive buying when price is trending up
                price_trend_factor = self.price_trends.get(product, {}).get("trend", 0)
                adjusted_fair = fair_value * (1 + price_trend_factor * 0.01)
                
                # Buy if price is favorable
                if ask_price < adjusted_fair - (dynamic_spread * buy_appetite):
                    # Calculate quantity to buy based on position limits
                    max_buy = position_limit - position
                    buy_quantity = min(max_buy, ask_volume, int(30 * buy_appetite))
                    
                    if buy_quantity > 0:
                        orders.append(Order(product, ask_price, buy_quantity))
                        position += buy_quantity  # Update position for next order
        
        # Sell orders - reduce selling as we approach short position limit
        sell_appetite = 1.0 - max(0, -position_ratio)
        
        if order_depth.buy_orders and position > -position_limit * 0.9:
            best_bids = sorted(order_depth.buy_orders.keys(), reverse=True)
            for bid_price in best_bids:
                bid_volume = order_depth.buy_orders[bid_price]
                
                # Factor in price trend for more aggressive selling when price is trending down
                price_trend_factor = self.price_trends.get(product, {}).get("trend", 0)
                adjusted_fair = fair_value * (1 - price_trend_factor * 0.01)
                
                # Sell if price is favorable
                if bid_price > adjusted_fair + (dynamic_spread * sell_appetite):
                    # Calculate quantity to sell based on position limits
                    max_sell = position_limit + position
                    sell_quantity = min(max_sell, bid_volume, int(30 * sell_appetite))
                    
                    if sell_quantity > 0:
                        orders.append(Order(product, bid_price, -sell_quantity))
                        position -= sell_quantity  # Update position for next order
        
        return orders
    
    def handle_conversions(self, state):
        """
        Handle basket conversions based on position and market conditions.
        Returns a conversion value for either PICNIC_BASKET1 or PICNIC_BASKET2.
        """
        # Check if we have excess baskets that should be converted to components
        basket1_pos = state.position.get(Product.PICNIC_BASKET1, 0)
        basket2_pos = state.position.get(Product.PICNIC_BASKET2, 0)
        
        # If we don't have any baskets, we can't convert anything
        if basket1_pos <= 0 and basket2_pos <= 0:
            return 0
            
        # If we have both baskets, decide which is more profitable to convert
        if basket1_pos > 0 and basket2_pos > 0:
            # Determine which conversion is more profitable based on component values
            basket1_profit = self.calculate_conversion_profit(Product.PICNIC_BASKET1, state)
            basket2_profit = self.calculate_conversion_profit(Product.PICNIC_BASKET2, state)
            
            if basket1_profit > basket2_profit and basket1_profit > 0:
                conversion_basket = Product.PICNIC_BASKET1
                conversion_count = min(basket1_pos, 5)  # Maximum 5 conversions per iteration
            elif basket2_profit > 0:
                conversion_basket = Product.PICNIC_BASKET2
                conversion_count = min(basket2_pos, 5)  # Maximum 5 conversions per iteration
            else:
                return 0  # No profitable conversions
                
        # If we only have BASKET1
        elif basket1_pos > 0:
            basket1_profit = self.calculate_conversion_profit(Product.PICNIC_BASKET1, state)
            if basket1_profit > 0:
                conversion_basket = Product.PICNIC_BASKET1
                conversion_count = min(basket1_pos, 5)
            else:
                return 0
                
        # If we only have BASKET2
        else:  # basket2_pos > 0
            basket2_profit = self.calculate_conversion_profit(Product.PICNIC_BASKET2, state)
            if basket2_profit > 0:
                conversion_basket = Product.PICNIC_BASKET2
                conversion_count = min(basket2_pos, 5)
            else:
                return 0
        
        # Track this pending conversion
        self.pending_conversions = {
            "basket": conversion_basket,
            "count": conversion_count,
            "timestamp": state.timestamp
        }
        
        print(f"Requesting conversion of {conversion_count} {conversion_basket}")
        
        # Return the conversion count
        return conversion_count
    
    def calculate_conversion_profit(self, basket, state):
        """Calculate the profit from converting a basket to its components"""
        if basket not in self.basket_contents:
            return 0
            
        # Get the components and their quantities
        components = self.basket_contents[basket]
        
        # Calculate the value of components
        component_value = 0
        for component, quantity in components.items():
            if component not in state.order_depths or not state.order_depths[component].buy_orders:
                return 0  # Can't calculate profit if we can't sell components
                
            best_bid = max(state.order_depths[component].buy_orders.keys())
            component_value += quantity * best_bid
        
        # Get the basket value
        if basket not in state.order_depths or not state.order_depths[basket].sell_orders:
            return 0  # Can't calculate profit if we can't buy more baskets
            
        best_ask = min(state.order_depths[basket].sell_orders.keys())
        
        # Calculate profit
        return component_value - best_ask
    
    def find_basket_arbitrage(self, state):
        """Find and execute arbitrage opportunities between baskets and components"""
        result = {}  # Always initialize with empty dict
        
        # Check for PICNIC_BASKET2 arbitrage (4 CROISSANTS + 2 JAMS)
        if all(p in state.order_depths for p in [Product.CROISSANTS, Product.JAMS, Product.PICNIC_BASKET2]):
            croissant_depth = state.order_depths[Product.CROISSANTS]
            jam_depth = state.order_depths[Product.JAMS]
            basket2_depth = state.order_depths[Product.PICNIC_BASKET2]
            
            # Check if we can buy components and sell basket for profit
            if croissant_depth.sell_orders and jam_depth.sell_orders and basket2_depth.buy_orders:
                best_croissant_ask = min(croissant_depth.sell_orders.keys())
                best_jam_ask = min(jam_depth.sell_orders.keys())
                best_basket2_bid = max(basket2_depth.buy_orders.keys())
                
                # Cost to buy components
                component_cost = (4 * best_croissant_ask + 2 * best_jam_ask)
                
                # If we can make profit (reduced profit threshold to 0.5%)
                if best_basket2_bid > component_cost * 1.005:
                    profit_per_unit = best_basket2_bid - component_cost
                    print(f"BASKET2 Arbitrage: Buy components for {component_cost}, sell basket for {best_basket2_bid}, profit: {profit_per_unit}")
                    
                    # Calculate how many complete sets we can trade
                    croissant_pos = state.position.get(Product.CROISSANTS, 0)
                    jam_pos = state.position.get(Product.JAMS, 0)
                    basket2_pos = state.position.get(Product.PICNIC_BASKET2, 0)
                    
                    max_croissants = min(
                        self.position_limits[Product.CROISSANTS] - croissant_pos,
                        abs(croissant_depth.sell_orders[best_croissant_ask])
                    ) // 4
                    
                    max_jams = min(
                        self.position_limits[Product.JAMS] - jam_pos,
                        abs(jam_depth.sell_orders[best_jam_ask])
                    ) // 2
                    
                    max_baskets = min(
                        self.position_limits[Product.PICNIC_BASKET2] + basket2_pos,
                        basket2_depth.buy_orders[best_basket2_bid]
                    )
                    
                    # Calculate arbitrage quantity, increased from 10 to 20 for more profit
                    arb_quantity = min(max_croissants, max_jams, max_baskets, 20)
                    
                    if arb_quantity > 0:
                        # Buy components
                        result[Product.CROISSANTS] = [Order(Product.CROISSANTS, best_croissant_ask, 4 * arb_quantity)]
                        result[Product.JAMS] = [Order(Product.JAMS, best_jam_ask, 2 * arb_quantity)]
                        
                        # Sell basket
                        result[Product.PICNIC_BASKET2] = [Order(Product.PICNIC_BASKET2, best_basket2_bid, -arb_quantity)]
            
            # Check if we can buy basket and sell components for profit
            if croissant_depth.buy_orders and jam_depth.buy_orders and basket2_depth.sell_orders:
                best_croissant_bid = max(croissant_depth.buy_orders.keys())
                best_jam_bid = max(jam_depth.buy_orders.keys())
                best_basket2_ask = min(basket2_depth.sell_orders.keys())
                
                # Revenue from selling components
                component_revenue = (4 * best_croissant_bid + 2 * best_jam_bid)
                
                # If we can make profit (reduced profit threshold to 0.5%)
                if component_revenue > best_basket2_ask * 1.005:
                    profit_per_unit = component_revenue - best_basket2_ask
                    print(f"BASKET2 Arbitrage: Buy basket for {best_basket2_ask}, sell components for {component_revenue}, profit: {profit_per_unit}")
                    
                    # Calculate how many complete sets we can trade
                    croissant_pos = state.position.get(Product.CROISSANTS, 0)
                    jam_pos = state.position.get(Product.JAMS, 0)
                    basket2_pos = state.position.get(Product.PICNIC_BASKET2, 0)
                    
                    max_croissants = min(
                        self.position_limits[Product.CROISSANTS] + croissant_pos,
                        croissant_depth.buy_orders[best_croissant_bid]
                    ) // 4
                    
                    max_jams = min(
                        self.position_limits[Product.JAMS] + jam_pos,
                        jam_depth.buy_orders[best_jam_bid]
                    ) // 2
                    
                    max_baskets = min(
                        self.position_limits[Product.PICNIC_BASKET2] - basket2_pos,
                        abs(basket2_depth.sell_orders[best_basket2_ask])
                    )
                    
                    # Calculate arbitrage quantity, increased from 10 to 20 for more profit
                    arb_quantity = min(max_croissants, max_jams, max_baskets, 20)
                    
                    if arb_quantity > 0:
                        # Buy basket
                        result[Product.PICNIC_BASKET2] = [Order(Product.PICNIC_BASKET2, best_basket2_ask, arb_quantity)]
                        
                        # Sell components
                        result[Product.CROISSANTS] = [Order(Product.CROISSANTS, best_croissant_bid, -4 * arb_quantity)]
                        result[Product.JAMS] = [Order(Product.JAMS, best_jam_bid, -2 * arb_quantity)]
        
        # Check for PICNIC_BASKET1 arbitrage (6 CROISSANTS + 3 JAMS + 1 DJEMBES)
        if all(p in state.order_depths for p in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES, Product.PICNIC_BASKET1]):
            croissant_depth = state.order_depths[Product.CROISSANTS]
            jam_depth = state.order_depths[Product.JAMS]
            djembe_depth = state.order_depths[Product.DJEMBES]
            basket1_depth = state.order_depths[Product.PICNIC_BASKET1]
            
            # Check if we can buy components and sell basket for profit
            if (croissant_depth.sell_orders and jam_depth.sell_orders and 
                djembe_depth.sell_orders and basket1_depth.buy_orders):
                best_croissant_ask = min(croissant_depth.sell_orders.keys())
                best_jam_ask = min(jam_depth.sell_orders.keys())
                best_djembe_ask = min(djembe_depth.sell_orders.keys())
                best_basket1_bid = max(basket1_depth.buy_orders.keys())
                
                # Cost to buy components
                component_cost = (6 * best_croissant_ask + 3 * best_jam_ask + 1 * best_djembe_ask)
                
                # If we can make profit (reduced profit threshold to 0.5%)
                if best_basket1_bid > component_cost * 1.005:
                    profit_per_unit = best_basket1_bid - component_cost
                    print(f"BASKET1 Arbitrage: Buy components for {component_cost}, sell basket for {best_basket1_bid}, profit: {profit_per_unit}")
                    
                    # Calculate how many complete sets we can trade
                    croissant_pos = state.position.get(Product.CROISSANTS, 0)
                    jam_pos = state.position.get(Product.JAMS, 0)
                    djembe_pos = state.position.get(Product.DJEMBES, 0)
                    basket1_pos = state.position.get(Product.PICNIC_BASKET1, 0)
                    
                    max_croissants = min(
                        self.position_limits[Product.CROISSANTS] - croissant_pos,
                        abs(croissant_depth.sell_orders[best_croissant_ask])
                    ) // 6
                    
                    max_jams = min(
                        self.position_limits[Product.JAMS] - jam_pos,
                        abs(jam_depth.sell_orders[best_jam_ask])
                    ) // 3
                    
                    max_djembes = min(
                        self.position_limits[Product.DJEMBES] - djembe_pos,
                        abs(djembe_depth.sell_orders[best_djembe_ask])
                    )
                    
                    max_baskets = min(
                        self.position_limits[Product.PICNIC_BASKET1] + basket1_pos,
                        basket1_depth.buy_orders[best_basket1_bid]
                    )
                    
                    # Calculate arbitrage quantity, increased from 5 to 10 for more profit
                    arb_quantity = min(max_croissants, max_jams, max_djembes, max_baskets, 10)
                    
                    if arb_quantity > 0:
                        # Buy components
                        result[Product.CROISSANTS] = [Order(Product.CROISSANTS, best_croissant_ask, 6 * arb_quantity)]
                        result[Product.JAMS] = [Order(Product.JAMS, best_jam_ask, 3 * arb_quantity)]
                        result[Product.DJEMBES] = [Order(Product.DJEMBES, best_djembe_ask, arb_quantity)]
                        
                        # Sell basket
                        result[Product.PICNIC_BASKET1] = [Order(Product.PICNIC_BASKET1, best_basket1_bid, -arb_quantity)]
            
            # Check if we can buy basket and sell components for profit
            if (croissant_depth.buy_orders and jam_depth.buy_orders and 
                djembe_depth.buy_orders and basket1_depth.sell_orders):
                best_croissant_bid = max(croissant_depth.buy_orders.keys())
                best_jam_bid = max(jam_depth.buy_orders.keys())
                best_djembe_bid = max(djembe_depth.buy_orders.keys())
                best_basket1_ask = min(basket1_depth.sell_orders.keys())
                
                # Revenue from selling components
                component_revenue = (6 * best_croissant_bid + 3 * best_jam_bid + 1 * best_djembe_bid)
                
                # If we can make profit (reduced profit threshold to 0.5%)
                if component_revenue > best_basket1_ask * 1.005:
                    profit_per_unit = component_revenue - best_basket1_ask
                    print(f"BASKET1 Arbitrage: Buy basket for {best_basket1_ask}, sell components for {component_revenue}, profit: {profit_per_unit}")
                    
                    # Calculate how many complete sets we can trade
                    croissant_pos = state.position.get(Product.CROISSANTS, 0)
                    jam_pos = state.position.get(Product.JAMS, 0)
                    djembe_pos = state.position.get(Product.DJEMBES, 0)
                    basket1_pos = state.position.get(Product.PICNIC_BASKET1, 0)
                    
                    max_croissants = min(
                        self.position_limits[Product.CROISSANTS] + croissant_pos,
                        croissant_depth.buy_orders[best_croissant_bid]
                    ) // 6
                    
                    max_jams = min(
                        self.position_limits[Product.JAMS] + jam_pos,
                        jam_depth.buy_orders[best_jam_bid]
                    ) // 3
                    
                    max_djembes = min(
                        self.position_limits[Product.DJEMBES] + djembe_pos,
                        djembe_depth.buy_orders[best_djembe_bid]
                    )
                    
                    max_baskets = min(
                        self.position_limits[Product.PICNIC_BASKET1] - basket1_pos,
                        abs(basket1_depth.sell_orders[best_basket1_ask])
                    )
                    
                    # Calculate arbitrage quantity, increased from 5 to 10 for more profit
                    arb_quantity = min(max_croissants, max_jams, max_djembes, max_baskets, 10)
                    
                    if arb_quantity > 0:
                        # Buy basket
                        result[Product.PICNIC_BASKET1] = [Order(Product.PICNIC_BASKET1, best_basket1_ask, arb_quantity)]
                        
                        # Sell components
                        result[Product.CROISSANTS] = [Order(Product.CROISSANTS, best_croissant_bid, -6 * arb_quantity)]
                        result[Product.JAMS] = [Order(Product.JAMS, best_jam_bid, -3 * arb_quantity)]
                        result[Product.DJEMBES] = [Order(Product.DJEMBES, best_djembe_bid, -arb_quantity)]
        
        return result
        
    def __init__(self, resin_kelp_params=None):
        # Volcanic Rock Voucher-related initialization
        self.position_limits = {
            # Volcanic rock products
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
            
            # Rainforest resin and kelp products
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            
            # Picnic basket products
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            
            # Squid ink product
            Product.SQUID_INK: 50,
            
            # Macarons product
            Product.MAGNIFICENT_MACARONS: 75
        }
        
        self.strike_prices = {
            Product.VOLCANIC_ROCK_VOUCHER_9500: 9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 10500
        }
        
        # Rainforest resin and kelp initialization
        if resin_kelp_params is None:
            resin_kelp_params = RESIN_KELP_PARAMS
        self.resin_kelp_params = resin_kelp_params
        
        # Picnic basket initialization
        self.fair_values = {}
        self.price_trends = {}
        self.spreads = {}
        self.positions = {}
        self.basket_contents = {
            Product.PICNIC_BASKET1: {Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1},
            Product.PICNIC_BASKET2: {Product.CROISSANTS: 4, Product.JAMS: 2, Product.DJEMBES: 0}
        }
        self.conversion_request = 0
        self.last_positions = {}
        self.pending_conversions = {}
        
        # Squid ink initialization
        self.squid_ink_state = SquidInkImbalanceState()
        
        # Macarons initialization
        self.macarons_state = MacaronsState()
    def run_volcanic_trading(self, state: TradingState):
        """Run the volcanic rock voucher trading strategy exactly as in original code"""
        result = {}
        conversions = 0
        # State for round tracking and underlying price history (not used for stdev)
        trader_data = {"underlying_prices": [], "round_number": 1, "prev_timestamp": None}
        
        if state.traderData:
            try:
                # Try to decode using jsonpickle (original method)
                trader_data = jsonpickle.decode(state.traderData)
            except:
                # If it fails, it might be in another format, but don't modify trader_data
                pass

        # Update round number based on timestamp changes
        if trader_data["prev_timestamp"] is not None and state.timestamp > trader_data["prev_timestamp"]:
            trader_data["round_number"] += 1
        trader_data["prev_timestamp"] = state.timestamp

        # Get underlying price from VOLCANIC_ROCK order book
        S = None
        if Product.VOLCANIC_ROCK in state.order_depths:
            underlying_book = state.order_depths[Product.VOLCANIC_ROCK]
            best_bid = max(underlying_book.buy_orders.keys()) if underlying_book.buy_orders else None
            best_ask = min(underlying_book.sell_orders.keys()) if underlying_book.sell_orders else None
            
            if best_bid is not None and best_ask is not None:
                S = (best_bid + best_ask) / 2
                trader_data["underlying_prices"].append(S)
                if len(trader_data["underlying_prices"]) > 20:
                    trader_data["underlying_prices"].pop(0)

        # Use a fixed volatility (e.g., 0.2) and risk-free rate (0.0)
        sigma = 0.31
        r = 0.0

        for product in state.order_depths:
            if "VOLCANIC_ROCK_VOUCHER" in product:
                order_depth = state.order_depths[product]
                orders = []
                current_position = state.position.get(product, 0)
                strike_price = self.strike_prices[product]
                
                # Calculate time to expiration (days_left = 7 - (round_number - 1))
                days_left = 7 - (trader_data["round_number"] - 1)
                T = max(days_left / 365.0, 1/365)  # At least 1 day

                if S and S > 0:
                    # Calculate fair value using Black-Scholes
                    fair_value = self.black_scholes(S, strike_price, T, r, sigma)
                    
                    # Market making logic with position limits
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    
                    # Buy logic
                    if best_ask is not None and best_ask < fair_value:
                        max_buy = self.position_limits[product] - current_position
                        if max_buy > 0:
                            ask_qty = min(-order_depth.sell_orders[best_ask], max_buy)
                            orders.append(Order(product, best_ask, ask_qty))
                            
                    # Sell logic
                    if best_bid is not None and best_bid > fair_value:
                        max_sell = self.position_limits[product] + current_position
                        if max_sell > 0:
                            bid_qty = min(order_depth.buy_orders[best_bid], max_sell)
                            orders.append(Order(product, best_bid, -bid_qty))
                            
                    result[product] = orders

        trader_data_str = jsonpickle.encode(trader_data)
        return result, conversions, trader_data_str
        
    def _norm_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function using math.erf."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def black_scholes(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0:
            return max(S - K, 0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self._norm_cdf(d1) - K * math.exp(-r * T) * self._norm_cdf(d2)
        
    def run(self, state: TradingState):
        # Initialize results
        result = {}
        
        # Split the execution of the trading strategies
        volcanic_result, volcanic_conversions, volcanic_trader_data = self.run_volcanic_trading(state)
        resin_kelp_result, resin_kelp_conversions, resin_kelp_trader_data = self.run_resin_kelp_trading(state)
        picnic_result, picnic_conversions, picnic_trader_data = self.run_picnic_trading(state)
        squid_result, squid_conversions, squid_trader_data = self.run_squid_trading(state)
        macarons_result, macarons_conversions, macarons_trader_data = self.run_macarons_trading(state)
        
        # Combine results
        result.update(volcanic_result)
        result.update(resin_kelp_result)
        result.update(picnic_result)
        result.update(squid_result)
        result.update(macarons_result)
        
        # Use conversion from picnic trader or macarons trader (prioritize picnic)
        conversions = picnic_conversions
        if conversions == 0 and macarons_conversions != 0:
            conversions = macarons_conversions
        
        # Use trader data from volcanic trader to maintain its state exactly
        trader_data_str = volcanic_trader_data
        
        return result, conversions, trader_data_str
        result = {}
        conversions = 0
        # State for round tracking and underlying price history (not used for stdev)
        trader_data = {"underlying_prices": [], "round_number": 1, "prev_timestamp": None}
        
        if state.traderData:
            try:
                # Try to decode using jsonpickle (original method)
                trader_data = jsonpickle.decode(state.traderData)
            except:
                # If it fails, it might be in another format, but don't modify trader_data
                pass

        # Update round number based on timestamp changes
        if trader_data["prev_timestamp"] is not None and state.timestamp > trader_data["prev_timestamp"]:
            trader_data["round_number"] += 1
        trader_data["prev_timestamp"] = state.timestamp

        # Get underlying price from VOLCANIC_ROCK order book
        S = None
        if Product.VOLCANIC_ROCK in state.order_depths:
            underlying_book = state.order_depths[Product.VOLCANIC_ROCK]
            best_bid = max(underlying_book.buy_orders.keys()) if underlying_book.buy_orders else None
            best_ask = min(underlying_book.sell_orders.keys()) if underlying_book.sell_orders else None
            
            if best_bid is not None and best_ask is not None:
                S = (best_bid + best_ask) / 2
                trader_data["underlying_prices"].append(S)
                if len(trader_data["underlying_prices"]) > 20:
                    trader_data["underlying_prices"].pop(0)

        # Use a fixed volatility (e.g., 0.2) and risk-free rate (0.0)
        sigma = 0.15
        r = 0.0

        for product in state.order_depths:
            if "VOLCANIC_ROCK_VOUCHER" in product:
                order_depth = state.order_depths[product]
                orders = []
                current_position = state.position.get(product, 0)
                strike_price = self.strike_prices[product]
                
                # Calculate time to expiration (days_left = 7 - (round_number - 1))
                days_left = 7 - (trader_data["round_number"] - 1)
                T = max(days_left / 365.0, 1/365)  # At least 1 day

                if S and S > 0:
                    # Calculate fair value using Black-Scholes
                    fair_value = self.black_scholes(S, strike_price, T, r, sigma)
                    
                    # Market making logic with position limits
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    
                    # Buy logic
                    if best_ask is not None and best_ask < fair_value:
                        max_buy = self.position_limits[product] - current_position
                        if max_buy > 0:
                            ask_qty = min(-order_depth.sell_orders[best_ask], max_buy)
                            orders.append(Order(product, best_ask, ask_qty))
                            
                    # Sell logic
                    if best_bid is not None and best_bid > fair_value:
                        max_sell = self.position_limits[product] + current_position
                        if max_sell > 0:
                            bid_qty = min(order_depth.buy_orders[best_bid], max_sell)
                            orders.append(Order(product, best_bid, -bid_qty))
                            
                    result[product] = orders

        trader_data_str = jsonpickle.encode(trader_data)
        return result, conversions, trader_data_str
    
    def run_resin_kelp_trading(self, state: TradingState):
        """Run the rainforest resin and kelp trading strategies"""
        result = {}
        conversions = 0
        resin_kelp_trader_data = {}
        
        # Parse resin/kelp trader data if it exists
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
                if isinstance(trader_data, dict) and "kelp_last_price" in trader_data:
                    resin_kelp_trader_data = trader_data
            except:
                # If there's an error parsing, just use empty dict
                pass
        
        # Handle Rainforest Resin
        if Product.RAINFOREST_RESIN in self.resin_kelp_params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.resin_kelp_params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.resin_kelp_params[Product.RAINFOREST_RESIN]["take_width"],
                    resin_position,
                )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.resin_kelp_params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.resin_kelp_params[Product.RAINFOREST_RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            resin_make_orders, _, _ = self.make_resin_orders(
                state.order_depths[Product.RAINFOREST_RESIN],
                self.resin_kelp_params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.resin_kelp_params[Product.RAINFOREST_RESIN]["volume_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (
                resin_take_orders + resin_clear_orders + resin_make_orders
            )

        # Handle Kelp
        if Product.KELP in self.resin_kelp_params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], resin_kelp_trader_data
            )
            
            if kelp_fair_value is not None:
                kelp_take_orders, buy_order_volume, sell_order_volume = (
                    self.take_orders(
                        Product.KELP,
                        state.order_depths[Product.KELP],
                        kelp_fair_value,
                        self.resin_kelp_params[Product.KELP]["take_width"],
                        kelp_position,
                        self.resin_kelp_params[Product.KELP]["prevent_adverse"],
                        self.resin_kelp_params[Product.KELP]["adverse_volume"],
                    )
                )
                kelp_clear_orders, buy_order_volume, sell_order_volume = (
                    self.clear_orders(
                        Product.KELP,
                        state.order_depths[Product.KELP],
                        kelp_fair_value,
                        self.resin_kelp_params[Product.KELP]["clear_width"],
                        kelp_position,
                        buy_order_volume,
                        sell_order_volume,
                    )
                )
                kelp_make_orders, _, _ = self.make_kelp_orders(
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.resin_kelp_params[Product.KELP]["min_edge"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
                result[Product.KELP] = (
                    kelp_take_orders + kelp_clear_orders + kelp_make_orders
                )
        
        trader_data_str = jsonpickle.encode(resin_kelp_trader_data)
        return result, conversions, trader_data_str