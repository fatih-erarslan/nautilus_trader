/*!
 * Tax calculation modules
 *
 * Implements all tax calculation methods:
 * - FIFO (First-In, First-Out)
 * - LIFO (Last-In, First-Out)
 * - HIFO (Highest-In, First-Out)
 * - Specific Identification
 * - Average Cost
 * Plus wash sale detection and cost basis adjustments
 */

pub mod calculator;
pub mod fifo;
pub mod lifo;
pub mod hifo;
pub mod wash_sale;
pub mod specific_id;
pub mod average_cost;
