/**
 * Logger.js
 * 
 * Centralized logging system for the experiment.
 * Captures all major events (generation metrics, organism events, etc.)
 * and provides export functionality to CSV/JSON.
 */

'use strict';

class Logger {
    constructor() {
        this.logs = [];
        this.csv_header_written = false;
    }

    /**
     * Add a log entry with timestamp
     * @param {string} source - module name (e.g., "GAManager", "AdvancedOrganism")
     * @param {string} message - log message
     * @param {object} data - optional data object to include
     */
    log(source, message, data = null) {
        const entry = {
            timestamp: new Date().toISOString(),
            source,
            message,
            data: data || {}
        };
        this.logs.push(entry);
        
        // Also log to console for real-time viewing
        console.log(`[${source}] ${message}`, data || '');
    }

    /**
     * Export logs as CSV
     * @returns {string} CSV formatted logs
     */
    toCSV() {
        if (this.logs.length === 0) return 'No logs recorded';

        const headers = ['timestamp', 'source', 'message', 'data_json'];
        let csv = headers.join(',') + '\n';

        for (const entry of this.logs) {
            const timestamp = entry.timestamp;
            const source = entry.source;
            const message = entry.message.replace(/,/g, ';').replace(/"/g, '""');
            const data_json = JSON.stringify(entry.data).replace(/"/g, '""');
            
            csv += `"${timestamp}","${source}","${message}","${data_json}"\n`;
        }

        return csv;
    }

    /**
     * Export logs as JSON
     * @returns {string} JSON formatted logs
     */
    toJSON() {
        return JSON.stringify(this.logs, null, 2);
    }

    /**
     * Download logs as file
     * @param {string} format - 'csv' or 'json'
     * @param {string} filename - output filename
     */
    download(format = 'csv', filename = null) {
        const content = format === 'csv' ? this.toCSV() : this.toJSON();
        const ext = format === 'csv' ? 'csv' : 'json';
        const name = filename || `experiment_logs_${new Date().toISOString().split('T')[0]}.${ext}`;

        const blob = new Blob([content], { type: format === 'csv' ? 'text/csv' : 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = name;
        a.click();
        URL.revokeObjectURL(url);
    }

    /**
     * Clear all logs
     */
    clear() {
        this.logs = [];
    }

    /**
     * Get log count
     */
    count() {
        return this.logs.length;
    }
}

// Export singleton instance
const logger = new Logger();

module.exports = logger;
