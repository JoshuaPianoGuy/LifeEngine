'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const { URL } = require('url');

const PORT = Number(process.env.PORT) || 3000;
const DIST_DIR = path.join(__dirname, 'dist');
const LOG_DIR = path.join(__dirname, 'logs');
const MAX_BODY_BYTES = 64 * 1024 * 1024;  // 64 MB — accommodates full-genome snapshots for all organisms
const ALLOWED_LOG_FILES = new Set([
    'life-engine-logs.csv',
    'generations.csv',
    'organisms.csv',
    'events.csv',
    'genome.csv'
]);
const runDirCache = new Map();

const mimeTypes = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon'
};

function send(res, status, body, headers = {}) {
    res.writeHead(status, headers);
    res.end(body);
}

function appendSnapshot({ condition, mode, csv, filename }, res) {
    const cond = condition === 'learning' ? 'learning' : 'evolution';
    const modeDir = (mode === 'frozen_pg' || mode === 'pure_rl') ? mode : 'standard';
    const outDir = getRunDir(cond, modeDir);
    const resolvedName = filename || 'life-engine-logs.csv';
    if (!ALLOWED_LOG_FILES.has(resolvedName)) {
        return send(res, 400, JSON.stringify({ ok: false, error: 'Invalid filename' }), {
            'Content-Type': 'application/json'
        });
    }
    const filePath = path.join(outDir, resolvedName);

    try {
        fs.mkdirSync(outDir, { recursive: true });
        const hasFile = fs.existsSync(filePath);
        const hasContent = hasFile && fs.statSync(filePath).size > 0;
        const prefix = hasContent ? '\n' : '';
        fs.appendFileSync(filePath, prefix + csv, 'utf8');
        send(res, 200, JSON.stringify({ ok: true }), { 'Content-Type': 'application/json' });
    } catch (err) {
        send(res, 500, JSON.stringify({ ok: false, error: err.message || String(err) }), {
            'Content-Type': 'application/json'
        });
    }
}

function appendLogFile({ condition, mode, filename, header, rows }, res) {
    const cond = condition === 'learning' ? 'learning' : 'evolution';
    const modeDir = (mode === 'frozen_pg' || mode === 'pure_rl') ? mode : 'standard';
    if (!ALLOWED_LOG_FILES.has(filename)) {
        return send(res, 400, JSON.stringify({ ok: false, error: 'Invalid filename' }), {
            'Content-Type': 'application/json'
        });
    }
    const outDir = getRunDir(cond, modeDir);
    const filePath = path.join(outDir, filename);

    try {
        fs.mkdirSync(outDir, { recursive: true });
        const hasFile = fs.existsSync(filePath);
        const hasContent = hasFile && fs.statSync(filePath).size > 0;
        const prefix = hasContent && rows ? '\n' : '';
        const headerText = !hasContent && header ? header + (rows ? '\n' : '') : '';
        fs.appendFileSync(filePath, prefix + headerText + (rows || ''), 'utf8');
        send(res, 200, JSON.stringify({ ok: true }), { 'Content-Type': 'application/json' });
    } catch (err) {
        send(res, 500, JSON.stringify({ ok: false, error: err.message || String(err) }), {
            'Content-Type': 'application/json'
        });
    }
}

function getRunDir(condition, modeDir) {
    const key = `${condition}/${modeDir}`;
    if (runDirCache.has(key)) return runDirCache.get(key);

    const autoDir = path.join(LOG_DIR, condition, modeDir, 'auto-run');
    fs.mkdirSync(autoDir, { recursive: true });
    const entries = fs.readdirSync(autoDir, { withFileTypes: true })
        .filter(d => d.isDirectory() && /^run_\d+$/.test(d.name))
        .map(d => parseInt(d.name.replace('run_', ''), 10))
        .filter(n => Number.isFinite(n));
    const next = entries.length > 0 ? Math.max(...entries) + 1 : 1;
    const runDir = path.join(autoDir, `run_${next}`);
    runDirCache.set(key, runDir);
    return runDir;
}

function handleJsonPost(req, res, handler) {
    let body = '';
    let size = 0;
    req.on('data', chunk => {
        size += chunk.length;
        if (size > MAX_BODY_BYTES) {
            send(res, 413, 'Payload too large');
            req.destroy();
            return;
        }
        body += chunk.toString('utf8');
    });
    req.on('end', () => {
        try {
            const data = JSON.parse(body || '{}');
            return handler(data, res);
        } catch (err) {
            return send(res, 400, JSON.stringify({ ok: false, error: 'Invalid JSON' }), {
                'Content-Type': 'application/json'
            });
        }
    });
}

function serveStatic(req, res) {
    const url = new URL(req.url, `http://${req.headers.host}`);
    let relPath = url.pathname === '/' ? '/index.html' : url.pathname;
    relPath = path.posix.normalize(relPath);
    const filePath = path.join(DIST_DIR, relPath);

    if (!filePath.startsWith(DIST_DIR)) {
        return send(res, 400, 'Bad request');
    }

    fs.stat(filePath, (err, stat) => {
        if (err || !stat.isFile()) {
            return send(res, 404, 'Not found');
        }
        const ext = path.extname(filePath).toLowerCase();
        const type = mimeTypes[ext] || 'application/octet-stream';
        res.writeHead(200, { 'Content-Type': type });
        fs.createReadStream(filePath).pipe(res);
    });
}

const server = http.createServer((req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);

    if (url.pathname === '/api/logs/snapshot' && req.method === 'POST') {
        return handleJsonPost(req, res, (data, res_) => {
            if (!data || typeof data.csv !== 'string' || data.csv.length === 0) {
                return send(res_, 400, JSON.stringify({ ok: false, error: 'Missing csv' }), {
                    'Content-Type': 'application/json'
                });
            }
            return appendSnapshot(data, res_);
        });
    }

    if (url.pathname === '/api/logs/append' && req.method === 'POST') {
        return handleJsonPost(req, res, (data, res_) => {
            if (!data.filename) {
                return send(res_, 400, JSON.stringify({ ok: false, error: 'Missing filename' }), {
                    'Content-Type': 'application/json'
                });
            }
            const rows = typeof data.rows === 'string' ? data.rows : '';
            const header = typeof data.header === 'string' ? data.header : '';
            if (!rows && !header) {
                return send(res_, 400, JSON.stringify({ ok: false, error: 'Missing rows' }), {
                    'Content-Type': 'application/json'
                });
            }
            return appendLogFile(data, res_);
        });
    }

    if (req.method !== 'GET' && req.method !== 'HEAD') {
        return send(res, 405, 'Method not allowed');
    }

    return serveStatic(req, res);
});

server.listen(PORT, () => {
    console.log(`[server] http://localhost:${PORT}`);
});