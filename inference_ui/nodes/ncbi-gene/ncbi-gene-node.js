// ============================================
// NCBI Gene Tool Node
// ============================================

NodeRegistry.register('ncbi_gene', {
    label: 'NCBI Gene',
    category: 'Tool',
    allowRef: true,

    ports: [
        { name: 'in', dir: 'in', type: 'any' },
        { name: 'out', dir: 'out', type: 'any' }
    ],

    defaultConfig: {
        title: 'NCBI Gene',
        tool: 'ncbi_gene',
        menuTag: {
            en: 'Gene DB', ko: '유전자 DB', ja: '遺伝子DB', zh: '基因数据库',
            fr: 'Base gènes', de: 'Gen-DB', es: 'BD Genes', it: 'DB Geni',
            pt: 'BD Genes', nl: 'Gen-DB', ru: 'БД генов', ar: 'قاعدة الجينات',
            hi: 'जीन DB', tr: 'Gen DB', pl: 'Baza genów', cs: 'Genová DB',
            sv: 'Gen-DB', da: 'Gen-DB', no: 'Gen-DB', fi: 'Geeni-DB',
            el: 'Γονιδιακή ΒΔ', hu: 'Gén DB', ro: 'BD Gene', uk: 'БД генів',
            vi: 'CSDL Gen', th: 'ฐานข้อมูลยีน', id: 'DB Gen'
        },
        description: {
            en: 'Query NCBI Gene database for gene information',
            ko: 'NCBI Gene 데이터베이스에서 유전자 정보 조회',
            ja: 'NCBI Geneデータベースで遺伝子情報を検索',
            zh: '查询NCBI Gene数据库获取基因信息',
            fr: 'Interroger la base de données NCBI Gene pour des informations génétiques',
            de: 'NCBI-Gene-Datenbank nach Geninformationen abfragen',
            es: 'Consultar la base de datos NCBI Gene para información genética',
            it: 'Interrogare il database NCBI Gene per informazioni genetiche',
            pt: 'Consultar o banco de dados NCBI Gene para informações genéticas',
            nl: 'NCBI Gene-database raadplegen voor genetische informatie',
            ru: 'Запрос к базе данных NCBI Gene для получения информации о генах',
            ar: 'استعلام قاعدة بيانات NCBI Gene للحصول على معلومات الجينات',
            hi: 'जीन जानकारी के लिए NCBI Gene डेटाबेस क्वेरी करें',
            tr: 'Gen bilgisi için NCBI Gene veritabanını sorgulayın',
            pl: 'Zapytaj bazę danych NCBI Gene o informacje o genach',
            cs: 'Dotazujte databázi NCBI Gene pro informace o genech',
            sv: 'Sök i NCBI Gene-databasen efter geninformation',
            da: 'Forespørg NCBI Gene-databasen om geninformation',
            no: 'Spør NCBI Gene-databasen om geninformasjon',
            fi: 'Hae geeni-informaatiota NCBI Gene -tietokannasta',
            el: 'Ερώτημα στη βάση δεδομένων NCBI Gene για πληροφορίες γονιδίων',
            hu: 'NCBI Gene adatbázis lekérdezése géninformációkért',
            ro: 'Interogați baza de date NCBI Gene pentru informații genetice',
            uk: 'Запит до бази даних NCBI Gene для отримання інформації про гени',
            vi: 'Truy vấn cơ sở dữ liệu NCBI Gene để tìm thông tin gen',
            th: 'สืบค้นฐานข้อมูล NCBI Gene เพื่อหาข้อมูลยีน',
            id: 'Kueri database NCBI Gene untuk informasi gen'
        },
        status: 'pending',
        stepNum: ''
    },

    render(node, helpers) {
        const el = document.createElement('div');
        el.innerHTML = `
            <div class="ng-ports-in-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-in" data-port-name="in" data-port-dir="in" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
            <div class="ng-node-header">
                <span class="ng-node-num">${helpers.escapeHtml(node.stepNum)}</span>
                <span class="ng-node-title">${helpers.escapeHtml(node.title)}</span>
            </div>
            <div class="ng-node-body">${helpers.escapeHtml(node.tool)}</div>
            <div class="ng-node-progress"></div>
            <div class="ng-ports-out-row" data-node-id="${node.id}">
                <div class="ng-port ng-port-out" data-port-name="out" data-port-dir="out" data-port-type="any" data-node-id="${node.id}"></div>
            </div>
        `;
        return el;
    },

    getDragHandle(el) { return el.querySelector('.ng-node-header'); }
});
