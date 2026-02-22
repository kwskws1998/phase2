// ============================================
// Node Type Registry
// ============================================

const NodeRegistry = {
    _types: {},

    register(type, definition) {
        if (!type || !definition) return;
        if (!definition.render || typeof definition.render !== 'function') {
            console.warn(`NodeRegistry: type "${type}" missing render(). Skipped.`);
            return;
        }
        definition.category = definition.category || 'General';
        definition.label = definition.label || type;
        definition.ports = definition.ports || [
            { name: 'in', dir: 'in', type: 'any' },
            { name: 'out', dir: 'out', type: 'any' }
        ];
        this._types[type] = definition;
    },

    get(type) {
        return this._types[type] || this._types['step'] || null;
    },

    getAll() {
        return Object.keys(this._types);
    },

    getByCategory() {
        const result = {};
        for (const [type, def] of Object.entries(this._types)) {
            const cat = def.category || 'General';
            if (!result[cat]) result[cat] = [];
            result[cat].push({ type, label: def.label || type, definition: def });
        }
        return result;
    },

    async loadAll() {
        const resp = await fetch('/api/node-manifest');
        if (!resp.ok) {
            console.error('NodeRegistry: failed to fetch node manifest');
            return;
        }
        const { files } = await resp.json();
        const load = (src) => new Promise((resolve) => {
            if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
            const s = document.createElement('script');
            s.src = src;
            s.onload = resolve;
            s.onerror = () => { console.warn(`NodeRegistry: failed to load ${src}`); resolve(); };
            document.head.appendChild(s);
        });
        const helpers = files.filter(f => f.includes('-helper') || f.includes('-util'));
        const nodes = files.filter(f => !f.includes('-helper') && !f.includes('-util'));
        for (const src of helpers) await load(src);
        await Promise.all(nodes.map(load));
    }
};
