// ============================================
// Subtract Node Type (Math category)
// ============================================

NodeRegistry.register('math_subtract', {
    label: 'Subtract',
    category: 'Math',

    ports: [
        { name: 'a', dir: 'in', type: 'numeric', label: 'A', defaultValue: 0 },
        { name: 'b', dir: 'in', type: 'numeric', label: 'B', defaultValue: 0 },
        { name: 'out', dir: 'out', type: 'numeric', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Subtract',
        status: 'pending',
        portValues: { a: 0, b: 0 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'a', label: 'A', type: 'numeric', defaultValue: 0 },
            { name: 'b', label: 'B', type: 'numeric', defaultValue: 0 }
        ], 'numeric');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
