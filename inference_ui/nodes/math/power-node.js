// ============================================
// Power Node Type (Math category)
// ============================================

NodeRegistry.register('math_power', {
    label: 'Power',
    category: 'Math',

    ports: [
        { name: 'base', dir: 'in', type: 'numeric', label: 'Base', defaultValue: 2 },
        { name: 'exp', dir: 'in', type: 'numeric', label: 'Exp', defaultValue: 2 },
        { name: 'out', dir: 'out', type: 'numeric', label: 'Result' }
    ],

    defaultConfig: {
        title: 'Power',
        menuTag: { en: 'Math', ko: '거듭제곱', ja: '数学', zh: '数学', fr: 'Math', de: 'Mathe', es: 'Matemáticas', it: 'Matematica', pt: 'Matemática', nl: 'Wiskunde', ru: 'Математика', ar: 'رياضيات', hi: 'गणित', tr: 'Matematik', pl: 'Matematyka', cs: 'Matematika', sv: 'Matematik', da: 'Matematik', no: 'Matematikk', fi: 'Matematiikka', el: 'Μαθηματικά', hu: 'Matematika', ro: 'Matematică', uk: 'Математика', vi: 'Toán học', th: 'คณิตศาสตร์', id: 'Matematika' },
        description: { en: 'Raise base to exponent (base^exp)', ko: '밑을 지수만큼 거듭제곱 (base^exp)', ja: '底を指数で累乗 (base^exp)', zh: '底数的指数次幂 (base^exp)', fr: 'Élever la base à la puissance (base^exp)', de: 'Basis potenzieren (base^exp)', es: 'Elevar base al exponente (base^exp)', it: 'Elevare la base alla potenza (base^exp)', pt: 'Elevar base ao expoente (base^exp)', nl: 'Grondtal tot macht verheffen (base^exp)', ru: 'Возведение основания в степень (base^exp)', ar: 'رفع الأساس إلى الأس (base^exp)', hi: 'आधार को घातांक तक बढ़ाएँ (base^exp)', tr: 'Tabanı üsse yükselt (base^exp)', pl: 'Podnieś podstawę do potęgi (base^exp)', cs: 'Umocnit základ na exponent (base^exp)', sv: 'Höj basen till exponenten (base^exp)', da: 'Ophøj grundtal til eksponent (base^exp)', no: 'Opphøy grunntall til eksponent (base^exp)', fi: 'Korota kantaluku eksponenttiin (base^exp)', el: 'Ύψωση βάσης σε εκθέτη (base^exp)', hu: 'Alap hatványozása (base^exp)', ro: 'Ridicare la putere (base^exp)', uk: 'Піднести основу до степеня (base^exp)', vi: 'Nâng cơ số lên lũy thừa (base^exp)', th: 'ยกกำลังฐานด้วยเลขชี้กำลัง (base^exp)', id: 'Pangkatkan basis ke eksponen (base^exp)' },
        status: 'pending',
        portValues: { base: 2, exp: 2 }
    },

    render(node, helpers) {
        return MathNodeHelper.render(node, helpers, [
            { name: 'base', label: 'Base', type: 'numeric', defaultValue: 2 },
            { name: 'exp', label: 'Exp', type: 'numeric', defaultValue: 2 }
        ], 'numeric');
    },

    getDragHandle(el) {
        return el.querySelector('.ng-node-header');
    }
});
