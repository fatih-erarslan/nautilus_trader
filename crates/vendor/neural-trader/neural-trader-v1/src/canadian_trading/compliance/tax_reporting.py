"""
Canadian Tax Reporting Module
Implements CRA tax reporting requirements including T5008 slips, capital gains tracking,
foreign income reporting, ACB calculations, and multi-currency tax handling.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import uuid
from collections import defaultdict
import csv
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class TaxYear:
    """Canadian tax year utilities"""
    
    @staticmethod
    def get_current_tax_year() -> int:
        """Get current Canadian tax year"""
        now = datetime.now()
        return now.year if now.month <= 12 else now.year
    
    @staticmethod
    def get_tax_year_bounds(year: int) -> Tuple[datetime, datetime]:
        """Get start and end dates for tax year"""
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31, 23, 59, 59)
        return start, end


class SecurityDisposition(Enum):
    """Types of security dispositions for tax purposes"""
    SALE = "sale"
    DEEMED_DISPOSITION = "deemed_disposition"
    GIFT = "gift"
    DEATH = "death"
    EMIGRATION = "emigration"


@dataclass
class T5008Slip:
    """T5008 Statement of Securities Transactions"""
    slip_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tax_year: int = field(default_factory=TaxYear.get_current_tax_year)
    recipient_name: str = ""
    recipient_sin: str = ""
    recipient_address: Dict[str, str] = field(default_factory=dict)
    
    # Box numbers for T5008
    box_16_proceeds: Decimal = Decimal("0.00")  # Proceeds of disposition
    box_17_security_name: str = ""
    box_18_security_symbol: str = ""
    box_19_quantity: int = 0
    box_20_acb: Optional[Decimal] = None  # Cost or book value
    box_21_outlays: Decimal = Decimal("0.00")  # Outlays and expenses
    box_22_gain_loss: Optional[Decimal] = None  # Capital gain/loss
    
    # Additional fields
    cusip_isin: Optional[str] = None
    trade_date: Optional[datetime] = None
    settlement_date: Optional[datetime] = None
    currency: str = "CAD"
    exchange_rate: Decimal = Decimal("1.00")
    
    def to_cra_xml(self) -> ET.Element:
        """Convert to CRA XML format for electronic filing"""
        slip = ET.Element("T5008Slip")
        
        # Recipient information
        recipient = ET.SubElement(slip, "Recipient")
        ET.SubElement(recipient, "Name").text = self.recipient_name
        ET.SubElement(recipient, "SIN").text = self.recipient_sin.replace(" ", "")
        
        address = ET.SubElement(recipient, "Address")
        ET.SubElement(address, "Line1").text = self.recipient_address.get("line1", "")
        ET.SubElement(address, "City").text = self.recipient_address.get("city", "")
        ET.SubElement(address, "Province").text = self.recipient_address.get("province", "")
        ET.SubElement(address, "PostalCode").text = self.recipient_address.get("postal_code", "")
        
        # Transaction details
        transaction = ET.SubElement(slip, "Transaction")
        ET.SubElement(transaction, "Box16_Proceeds").text = str(self.box_16_proceeds)
        ET.SubElement(transaction, "Box17_SecurityName").text = self.box_17_security_name
        ET.SubElement(transaction, "Box18_Symbol").text = self.box_18_security_symbol
        ET.SubElement(transaction, "Box19_Quantity").text = str(self.box_19_quantity)
        
        if self.box_20_acb is not None:
            ET.SubElement(transaction, "Box20_ACB").text = str(self.box_20_acb)
        
        ET.SubElement(transaction, "Box21_Outlays").text = str(self.box_21_outlays)
        
        if self.box_22_gain_loss is not None:
            ET.SubElement(transaction, "Box22_GainLoss").text = str(self.box_22_gain_loss)
        
        if self.cusip_isin:
            ET.SubElement(transaction, "CUSIP_ISIN").text = self.cusip_isin
        
        return slip


@dataclass
class ACBTracker:
    """Adjusted Cost Base tracker for Canadian tax purposes"""
    symbol: str
    total_shares: int = 0
    total_acb: Decimal = Decimal("0.00")
    currency: str = "CAD"
    transactions: List[Dict] = field(default_factory=list)
    
    def add_purchase(self, shares: int, total_cost: Decimal, 
                    commission: Decimal, trade_date: datetime,
                    exchange_rate: Decimal = Decimal("1.00")) -> None:
        """Add purchase to ACB calculation"""
        # Convert to CAD if necessary
        cad_cost = (total_cost + commission) * exchange_rate
        
        self.total_shares += shares
        self.total_acb += cad_cost
        
        self.transactions.append({
            'type': 'purchase',
            'date': trade_date,
            'shares': shares,
            'total_cost': total_cost,
            'commission': commission,
            'exchange_rate': exchange_rate,
            'cad_cost': cad_cost,
            'new_acb': self.total_acb,
            'new_shares': self.total_shares,
            'acb_per_share': self.get_acb_per_share()
        })
    
    def add_sale(self, shares: int, proceeds: Decimal, 
                commission: Decimal, trade_date: datetime,
                exchange_rate: Decimal = Decimal("1.00")) -> Dict[str, Any]:
        """Calculate capital gain/loss for sale"""
        if shares > self.total_shares:
            raise ValueError(f"Cannot sell {shares} shares, only {self.total_shares} available")
        
        # Calculate ACB of shares sold
        acb_per_share = self.get_acb_per_share()
        acb_of_sold = acb_per_share * shares
        
        # Convert proceeds to CAD
        cad_proceeds = proceeds * exchange_rate
        cad_commission = commission * exchange_rate
        net_proceeds = cad_proceeds - cad_commission
        
        # Calculate capital gain/loss
        capital_gain = net_proceeds - acb_of_sold
        
        # Update holdings
        self.total_shares -= shares
        self.total_acb -= acb_of_sold
        
        transaction = {
            'type': 'sale',
            'date': trade_date,
            'shares': shares,
            'proceeds': proceeds,
            'commission': commission,
            'exchange_rate': exchange_rate,
            'cad_proceeds': cad_proceeds,
            'acb_per_share': acb_per_share,
            'acb_of_sold': acb_of_sold,
            'capital_gain': capital_gain,
            'new_acb': self.total_acb,
            'new_shares': self.total_shares
        }
        
        self.transactions.append(transaction)
        
        return {
            'capital_gain': capital_gain,
            'acb_used': acb_of_sold,
            'net_proceeds': net_proceeds,
            'remaining_shares': self.total_shares,
            'remaining_acb': self.total_acb
        }
    
    def add_return_of_capital(self, amount_per_share: Decimal, 
                             payment_date: datetime,
                             exchange_rate: Decimal = Decimal("1.00")) -> None:
        """Handle return of capital distributions"""
        total_roc = amount_per_share * self.total_shares * exchange_rate
        self.total_acb -= total_roc
        
        # ACB cannot go negative
        if self.total_acb < 0:
            capital_gain = abs(self.total_acb)
            self.total_acb = Decimal("0.00")
        else:
            capital_gain = Decimal("0.00")
        
        self.transactions.append({
            'type': 'return_of_capital',
            'date': payment_date,
            'amount_per_share': amount_per_share,
            'total_amount': total_roc,
            'exchange_rate': exchange_rate,
            'capital_gain': capital_gain,
            'new_acb': self.total_acb
        })
    
    def add_stock_dividend(self, shares: int, fair_market_value: Decimal,
                          payment_date: datetime,
                          exchange_rate: Decimal = Decimal("1.00")) -> None:
        """Handle stock dividends"""
        cad_value = fair_market_value * exchange_rate
        
        self.total_shares += shares
        self.total_acb += cad_value
        
        self.transactions.append({
            'type': 'stock_dividend',
            'date': payment_date,
            'shares': shares,
            'fmv': fair_market_value,
            'exchange_rate': exchange_rate,
            'cad_value': cad_value,
            'new_acb': self.total_acb,
            'new_shares': self.total_shares
        })
    
    def get_acb_per_share(self) -> Decimal:
        """Calculate current ACB per share"""
        if self.total_shares == 0:
            return Decimal("0.00")
        return (self.total_acb / self.total_shares).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )
    
    def handle_stock_split(self, split_ratio: str, split_date: datetime) -> None:
        """Handle stock splits (e.g., "2:1" for 2-for-1 split)"""
        new_shares, old_shares = map(int, split_ratio.split(':'))
        multiplier = new_shares / old_shares
        
        self.total_shares = int(self.total_shares * multiplier)
        # ACB remains the same, but per-share ACB changes
        
        self.transactions.append({
            'type': 'stock_split',
            'date': split_date,
            'ratio': split_ratio,
            'new_shares': self.total_shares,
            'acb_per_share': self.get_acb_per_share()
        })


class TaxReporting:
    """Main tax reporting implementation for CRA compliance"""
    
    def __init__(self):
        self.acb_trackers: Dict[str, ACBTracker] = {}
        self.t5008_slips: List[T5008Slip] = []
        self.foreign_income: List[Dict] = []
        self.exchange_rates = self._initialize_exchange_rates()
        self.provincial_tax_rates = self._initialize_provincial_rates()
        
    def _initialize_exchange_rates(self) -> Dict[str, Decimal]:
        """Initialize exchange rates (in production, would fetch from BoC)"""
        return {
            'USD': Decimal("1.35"),  # USD to CAD
            'EUR': Decimal("1.45"),  # EUR to CAD
            'GBP': Decimal("1.70"),  # GBP to CAD
            'JPY': Decimal("0.0090"),  # JPY to CAD
        }
    
    def _initialize_provincial_rates(self) -> Dict[str, Dict]:
        """Initialize provincial tax rates and rules"""
        return {
            "ON": {
                "capital_gains_inclusion": 0.50,  # 50% inclusion rate
                "dividend_gross_up": 1.38,
                "dividend_tax_credit": 0.25
            },
            "QC": {
                "capital_gains_inclusion": 0.50,
                "dividend_gross_up": 1.38,
                "dividend_tax_credit": 0.2661,  # Quebec has different credit
                "additional_requirements": ["TP-1000 filing", "French documentation"]
            },
            "BC": {
                "capital_gains_inclusion": 0.50,
                "dividend_gross_up": 1.38,
                "dividend_tax_credit": 0.25
            },
            "AB": {
                "capital_gains_inclusion": 0.50,
                "dividend_gross_up": 1.38,
                "dividend_tax_credit": 0.25
            }
        }
    
    def process_trade_for_tax(self, trade: Dict) -> Dict[str, Any]:
        """Process a trade for tax reporting purposes"""
        symbol = trade['symbol']
        trade_type = trade['type']  # 'buy' or 'sell'
        
        # Initialize ACB tracker if needed
        if symbol not in self.acb_trackers:
            self.acb_trackers[symbol] = ACBTracker(symbol=symbol)
        
        tracker = self.acb_trackers[symbol]
        
        # Get exchange rate
        currency = trade.get('currency', 'CAD')
        exchange_rate = self.exchange_rates.get(currency, Decimal("1.00")) if currency != 'CAD' else Decimal("1.00")
        
        if trade_type == 'buy':
            tracker.add_purchase(
                shares=trade['quantity'],
                total_cost=Decimal(str(trade['price'])) * trade['quantity'],
                commission=Decimal(str(trade.get('commission', 0))),
                trade_date=datetime.fromisoformat(trade['trade_date']),
                exchange_rate=exchange_rate
            )
            
            return {
                'status': 'success',
                'action': 'purchase_recorded',
                'symbol': symbol,
                'new_acb': tracker.total_acb,
                'new_shares': tracker.total_shares,
                'acb_per_share': tracker.get_acb_per_share()
            }
            
        elif trade_type == 'sell':
            result = tracker.add_sale(
                shares=trade['quantity'],
                proceeds=Decimal(str(trade['price'])) * trade['quantity'],
                commission=Decimal(str(trade.get('commission', 0))),
                trade_date=datetime.fromisoformat(trade['trade_date']),
                exchange_rate=exchange_rate
            )
            
            # Generate T5008 slip if in taxable account
            if trade.get('account_type') not in ['TFSA', 'RRSP', 'RRIF', 'RESP']:
                self._generate_t5008(trade, result, tracker)
            
            return {
                'status': 'success',
                'action': 'sale_processed',
                'symbol': symbol,
                'capital_gain': result['capital_gain'],
                'tax_implications': self._calculate_tax_implications(result['capital_gain'], trade.get('province', 'ON')),
                't5008_generated': trade.get('account_type') not in ['TFSA', 'RRSP', 'RRIF', 'RESP']
            }
    
    def _generate_t5008(self, trade: Dict, sale_result: Dict, tracker: ACBTracker) -> T5008Slip:
        """Generate T5008 slip for securities transaction"""
        slip = T5008Slip(
            tax_year=datetime.fromisoformat(trade['trade_date']).year,
            recipient_name=trade.get('client_name', ''),
            recipient_sin=trade.get('client_sin', ''),
            recipient_address=trade.get('client_address', {}),
            box_16_proceeds=sale_result['net_proceeds'],
            box_17_security_name=trade.get('security_name', trade['symbol']),
            box_18_security_symbol=trade['symbol'],
            box_19_quantity=trade['quantity'],
            box_20_acb=sale_result['acb_used'],
            box_21_outlays=Decimal(str(trade.get('commission', 0))),
            box_22_gain_loss=sale_result['capital_gain'],
            cusip_isin=trade.get('cusip') or trade.get('isin'),
            trade_date=datetime.fromisoformat(trade['trade_date']),
            settlement_date=datetime.fromisoformat(trade.get('settlement_date', trade['trade_date'])),
            currency=trade.get('currency', 'CAD'),
            exchange_rate=self.exchange_rates.get(trade.get('currency', 'CAD'), Decimal("1.00"))
        )
        
        self.t5008_slips.append(slip)
        return slip
    
    def _calculate_tax_implications(self, capital_gain: Decimal, province: str) -> Dict[str, Any]:
        """Calculate tax implications of capital gain"""
        provincial_rules = self.provincial_tax_rates.get(province, self.provincial_tax_rates['ON'])
        
        taxable_capital_gain = capital_gain * Decimal(str(provincial_rules['capital_gains_inclusion']))
        
        # Estimate tax (in production, would use actual marginal rates)
        estimated_federal_tax = taxable_capital_gain * Decimal("0.15")  # Lowest federal bracket
        estimated_provincial_tax = taxable_capital_gain * Decimal("0.0505")  # ON lowest bracket
        
        return {
            'capital_gain': capital_gain,
            'taxable_capital_gain': taxable_capital_gain,
            'inclusion_rate': provincial_rules['capital_gains_inclusion'],
            'estimated_federal_tax': estimated_federal_tax,
            'estimated_provincial_tax': estimated_provincial_tax,
            'estimated_total_tax': estimated_federal_tax + estimated_provincial_tax,
            'province': province
        }
    
    def record_foreign_income(self, income: Dict) -> Dict[str, Any]:
        """Record foreign income for T1135 and other reporting"""
        income_record = {
            'income_id': str(uuid.uuid4()),
            'date': income['date'],
            'type': income['type'],  # dividend, interest, capital_gain
            'country': income['country'],
            'currency': income['currency'],
            'foreign_amount': Decimal(str(income['amount'])),
            'exchange_rate': self.exchange_rates.get(income['currency'], Decimal("1.00")),
            'cad_amount': Decimal(str(income['amount'])) * self.exchange_rates.get(income['currency'], Decimal("1.00")),
            'foreign_tax_paid': Decimal(str(income.get('foreign_tax', 0))),
            'foreign_tax_paid_cad': Decimal(str(income.get('foreign_tax', 0))) * self.exchange_rates.get(income['currency'], Decimal("1.00"))
        }
        
        self.foreign_income.append(income_record)
        
        # Check if T1135 reporting required
        t1135_required = self._check_t1135_requirement(income.get('client_id'))
        
        return {
            'status': 'success',
            'income_recorded': True,
            'cad_amount': income_record['cad_amount'],
            'foreign_tax_credit_available': income_record['foreign_tax_paid_cad'],
            't1135_required': t1135_required,
            'record_id': income_record['income_id']
        }
    
    def _check_t1135_requirement(self, client_id: str) -> bool:
        """Check if T1135 foreign asset reporting is required"""
        # T1135 required if cost of foreign property > $100,000 CAD
        # In production, would calculate actual foreign property cost
        total_foreign_cost = Decimal("0.00")
        
        for symbol, tracker in self.acb_trackers.items():
            if self._is_foreign_property(symbol):
                total_foreign_cost += tracker.total_acb
        
        return total_foreign_cost > Decimal("100000.00")
    
    def _is_foreign_property(self, symbol: str) -> bool:
        """Determine if security is foreign property for T1135"""
        # Canadian securities typically end with .TO, .V, .CN
        canadian_exchanges = ['.TO', '.V', '.CN', '.TSX', '.CSE']
        return not any(symbol.endswith(ext) for ext in canadian_exchanges)
    
    def generate_year_end_tax_package(self, client_id: str, tax_year: int) -> Dict[str, Any]:
        """Generate complete tax package for client"""
        start_date, end_date = TaxYear.get_tax_year_bounds(tax_year)
        
        # Collect all T5008 slips for client and year
        client_t5008s = [
            slip for slip in self.t5008_slips 
            if slip.tax_year == tax_year and slip.recipient_sin == client_id
        ]
        
        # Calculate total capital gains/losses
        total_capital_gains = sum(
            slip.box_22_gain_loss for slip in client_t5008s 
            if slip.box_22_gain_loss and slip.box_22_gain_loss > 0
        )
        
        total_capital_losses = sum(
            abs(slip.box_22_gain_loss) for slip in client_t5008s 
            if slip.box_22_gain_loss and slip.box_22_gain_loss < 0
        )
        
        # Net capital gain/loss
        net_capital_gain = total_capital_gains - total_capital_losses
        
        # Foreign income summary
        client_foreign_income = [
            fi for fi in self.foreign_income
            if start_date <= datetime.fromisoformat(fi['date']) <= end_date
        ]
        
        total_foreign_income = sum(fi['cad_amount'] for fi in client_foreign_income)
        total_foreign_tax = sum(fi['foreign_tax_paid_cad'] for fi in client_foreign_income)
        
        package = {
            'client_id': client_id,
            'tax_year': tax_year,
            'generation_date': datetime.now().isoformat(),
            
            'capital_gains_summary': {
                'total_dispositions': len(client_t5008s),
                'total_proceeds': sum(slip.box_16_proceeds for slip in client_t5008s),
                'total_acb': sum(slip.box_20_acb for slip in client_t5008s if slip.box_20_acb),
                'total_outlays': sum(slip.box_21_outlays for slip in client_t5008s),
                'gross_capital_gains': total_capital_gains,
                'gross_capital_losses': total_capital_losses,
                'net_capital_gain_loss': net_capital_gain,
                'taxable_capital_gain': net_capital_gain * Decimal("0.50") if net_capital_gain > 0 else Decimal("0.00")
            },
            
            'foreign_income_summary': {
                'total_foreign_income': total_foreign_income,
                'total_foreign_tax_paid': total_foreign_tax,
                'foreign_tax_credit_available': min(total_foreign_tax, total_foreign_income * Decimal("0.15")),
                'countries': list(set(fi['country'] for fi in client_foreign_income))
            },
            
            'forms_required': {
                'schedule_3': net_capital_gain != 0,
                't5008_count': len(client_t5008s),
                't1135_required': self._check_t1135_requirement(client_id),
                'foreign_tax_credit': total_foreign_tax > 0
            },
            
            't5008_slips': [slip.to_cra_xml() for slip in client_t5008s],
            
            'tax_planning_notes': self._generate_tax_planning_notes(
                net_capital_gain, total_foreign_income, client_id
            )
        }
        
        return package
    
    def _generate_tax_planning_notes(self, net_capital_gain: Decimal, 
                                    foreign_income: Decimal, 
                                    client_id: str) -> List[str]:
        """Generate tax planning recommendations"""
        notes = []
        
        if net_capital_gain < 0:
            notes.append(f"Capital loss of ${abs(net_capital_gain)} can be carried back 3 years or forward indefinitely")
        
        if net_capital_gain > 50000:
            notes.append("Consider charitable donations of appreciated securities to avoid capital gains tax")
        
        if foreign_income > 10000:
            notes.append("Ensure foreign tax credits are claimed to avoid double taxation")
        
        # Check for tax-loss harvesting opportunities
        for symbol, tracker in self.acb_trackers.items():
            if tracker.total_shares > 0:
                current_value = tracker.total_shares * Decimal("100")  # Placeholder market price
                unrealized_loss = current_value - tracker.total_acb
                if unrealized_loss < -1000:
                    notes.append(f"Consider tax-loss harvesting for {symbol} (unrealized loss: ${abs(unrealized_loss)})")
        
        return notes
    
    def export_tax_data(self, client_id: str, tax_year: int, format: str = 'json') -> str:
        """Export tax data in various formats"""
        package = self.generate_year_end_tax_package(client_id, tax_year)
        
        if format == 'json':
            return json.dumps(package, default=str, indent=2)
        
        elif format == 'csv':
            # Export T5008 data as CSV
            csv_data = []
            csv_data.append(['Trade Date', 'Symbol', 'Quantity', 'Proceeds', 'ACB', 'Gain/Loss'])
            
            for slip in package['t5008_slips']:
                csv_data.append([
                    slip.trade_date.isoformat() if slip.trade_date else '',
                    slip.box_18_security_symbol,
                    slip.box_19_quantity,
                    slip.box_16_proceeds,
                    slip.box_20_acb or '',
                    slip.box_22_gain_loss or ''
                ])
            
            return '\n'.join([','.join(map(str, row)) for row in csv_data])
        
        elif format == 'xml':
            # CRA XML format
            root = ET.Element("TaxReturn")
            ET.SubElement(root, "TaxYear").text = str(tax_year)
            ET.SubElement(root, "ClientID").text = client_id
            
            slips_element = ET.SubElement(root, "T5008Slips")
            for slip in self.t5008_slips:
                if slip.tax_year == tax_year and slip.recipient_sin == client_id:
                    slips_element.append(slip.to_cra_xml())
            
            return ET.tostring(root, encoding='unicode')
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def calculate_quarterly_installments(self, client_id: str, current_year: int) -> Dict[str, Any]:
        """Calculate quarterly tax installments based on previous years"""
        # Get previous 2 years of tax data
        prev_year_1 = self.generate_year_end_tax_package(client_id, current_year - 1)
        prev_year_2 = self.generate_year_end_tax_package(client_id, current_year - 2)
        
        # Calculate average tax liability
        prev_tax_1 = prev_year_1['capital_gains_summary']['taxable_capital_gain'] * Decimal("0.25")  # Estimate
        prev_tax_2 = prev_year_2['capital_gains_summary']['taxable_capital_gain'] * Decimal("0.25")  # Estimate
        
        # CRA installment options
        option_1 = prev_tax_1 / 4  # Prior year
        option_2 = (prev_tax_1 + prev_tax_2) / 8  # Average of 2 years
        
        return {
            'installment_required': prev_tax_1 > 3000,  # $3,000 threshold
            'quarterly_amount_option_1': option_1,
            'quarterly_amount_option_2': option_2,
            'recommended_option': 'option_1' if option_1 < option_2 else 'option_2',
            'due_dates': {
                'Q1': f"{current_year}-03-15",
                'Q2': f"{current_year}-06-15",
                'Q3': f"{current_year}-09-15",
                'Q4': f"{current_year}-12-15"
            }
        }