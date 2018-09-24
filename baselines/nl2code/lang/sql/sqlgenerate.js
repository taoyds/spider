/**
 * Created by kai on 2018/4/23.
 */
import 'babel-polyfill';
import {map, join, head, compose, curry, toUpper, prop, equals, isEmpty, F, isArrayLike, concat, __, pluck, contains} from 'ramda';

const INDENT = ' ';
const LINE_END = '\n';

// This allows calling a function recursivly based on node type. Some of the
// nodes have non-standard types and so we need to modify how we call the
// Generator.
const recurse = curry((Generator, n) => {
    switch (n.type) {
        case 'function':
            return Generator['function'](n);
        case 'module':
            return Generator.module(n);
        case 'assignment':
            return Generator.assignment(n);
        case 'event':
            return Generator.event(n);
        default:
            return Generator[n.type][n.variant](n);
    }
});

const mapr = compose(map, recurse);

const datatype = (n) => n.variant;
const returnNewLine = join('\n');
const joinList = join(', ');
const terminateStatements = map(concat(__, ';'));
const containsSelect = (s) => (s.indexOf('SELECT') !== -1);
const isOfFormat = (n) => compose(equals(n), prop('format'));

var Generator = {
    assignment : (n) => {
        const recurser = recurse(Generator);
        const target = recurser(n.target);
        const value = recurser(n.value);
        return `${target} = ${value}`;
    },
    statement : {
        list : (n) => {
            const recourseOverList = mapr(Generator);
            const statements = compose(returnNewLine, terminateStatements, recourseOverList);
            return statements(n.statement);
        },
        select : (n) => {
            const recurser = recurse(Generator);
            const recourseList = mapr(Generator);
            const argsList = compose(joinList, recourseList);

            var str = [''];
            if (n.with) {
                const withS = recourseList(n.with);
                const isRecursive = (n) => isArrayLike(n) ? compose(contains('recursive'), pluck('variant'))(n) : F;
                const w = isRecursive(n.with) ? 'WITH RECURSIVE' : 'WITH';
                str.push(`${w} ${withS}${LINE_END}`);
            }
            str.push('SELECT ');
            if (n.result) {
                const results = argsList(n.result);
                str.push(`${results}${LINE_END}`);
            }
            if (n.from) {
                const from = recurser(n.from);
                str.push(`${INDENT}FROM ${from}${LINE_END}`);
            }
            if (n.where) {
                const where = recurser(head(n.where));
                str.push(`${INDENT}WHERE ${where}${LINE_END}`);
            }
            if (n.group) {
                const group = recurser(n.group);
                str.push(`${INDENT}GROUP BY ${group}${LINE_END}`);
            }
            if (n.having) {
                const having = recurser(n.having);
                str.push(`${INDENT}HAVING ${having}${LINE_END}`);
            }
            if (n.order) {
                const order = recourseList(n.order);
                str.push(`${INDENT}ORDER BY ${order}${LINE_END}`);
            }
            if (n.limit) {
                const limit = recurser(n.limit);
                str.push(`${INDENT}${limit}`);
            }
            return str.join('');
        },
        compound : (n) => {
            const recourseList = mapr(Generator);
            const statement = recurse(Generator)(n.statement);
            const compound = recourseList(n.compound).join('');
            const order = n.order ? `${INDENT}ORDER BY ${recourseList(n.order)}${LINE_END}` : '';
            return `${statement}${compound} ${order}`;
        },
        create : (n) => {
            const recurser = recurse(Generator);
            const isCreateIndex = isOfFormat('index');
            const isCreateTable = isOfFormat('table');
            const isCreateView = isOfFormat('view');
            const isCreateVirtual = isOfFormat('virtual');
            const isCreateTrigger = isOfFormat('trigger');

            if(isCreateTrigger(n)){
                const m = mapr(Generator);
                const target = recurser(n.target);
                const by = n.by ? `FOR EACH ${n.by}` : '';
                const event = recurser(n.event);
                const on = recurser(n.on);
                const action = compose(join(';\n'), m)(n.action);
                const when = recurser(n.when);
                const temporary = (!!n.temporary) ? 'TEMPORARY' : '';
                const condition = (n.condition) ? m(n.condition) : '';
                return `CREATE ${temporary} TRIGGER ${condition} ${target} ${event} ON ${on} ${by} WHEN ${when} BEGIN ${action}; END`;
            }

            if(isCreateVirtual(n)){
                const target = recurser(n.target);
                const result = recurser(n.result);
                return `CREATE VIRTUAL TABLE ${target} USING ${result}`;
            }

            if(isCreateView(n)){
                const viewName = recurser(n.target);
                const result = recurser(n.result);

                return `CREATE VIEW ${viewName}${LINE_END}AS ${result}`;
            }

            if (isCreateIndex(n)) {
                const indexName = n.target.name;
                const onColumns = recurser(n.on);
                const where = recurser(head(n.where));
                return `CREATE INDEX ${indexName}${LINE_END}ON ${onColumns}${LINE_END}WHERE ${where}`;
            }

            if (isCreateTable(n)) {
                const tableName = recurse(Generator)(n.name);
                const definitionsList = compose(join(`,${LINE_END}`), mapr(Generator));
                const definitions = definitionsList(n.definition);

                // Can probable be refactored to be a bit more elegant... :/
                const defaultCreateSyntax = `CREATE TABLE ${tableName} (${LINE_END}${definitions}${LINE_END})`;
                const createTableFromSelect = `CREATE TABLE ${tableName} AS${LINE_END}${definitions}${LINE_END}`;

                return containsSelect(definitions) ? createTableFromSelect
                                                   : defaultCreateSyntax;
            }
            return ``;
        },
        insert : (n) => {
            const recurser = recurse(Generator);
            const into = recurser(n.into);

            // This is an insert into default values
            if (n.result.variant === 'default'){
                return `INSERT INTO ${into}${LINE_END}DEFAULT VALUES`;
            }
            // This is an insert into select
            if (n.result.variant === 'select'){
                const result = recurser(n.result);
                return `INSERT INTO ${into}${LINE_END}${result}`;
            }
            // Otherwise we build up the values to be inserted
            const addBrackets = map((s) => `(${s})`);
            const valuesList = compose(join(`,${LINE_END}`), addBrackets, mapr(Generator));
            const result = valuesList(n.result);
            return `INSERT INTO ${into}${LINE_END}VALUES ${result}`;
        },
        'delete' : (n) => {
            const recurser = recurse(Generator);

            var str = ['DELETE '];

            if (n.from) {
                const from = recurser(n.from);
                str.push(`${INDENT}FROM ${from}${LINE_END}`);
            }
            if (n.where) {
                const whereNode = head(n.where);
                const where = recurser(whereNode);
                str.push(`${INDENT}WHERE ${where}${LINE_END}`);
            }
            if (n.limit) {
                const limit = recurser(n.limit);
                str.push(`${INDENT}${limit}`);
            }
            return str.join('');
        },
        drop : (n) => {
            const recurser = recurse(Generator);
            const condition = (n.condition.length > 0) ? mapr(Generator)(n.condition) : '';
            const target = recurser(n.target);
            return `DROP ${n.format} ${condition} ${target}`;
        },
        update : (n) => {
            const recurser = recurse(Generator);
            const into = recurser(n.into);
            const setS = mapr(Generator)(n.set);
            var str = [`UPDATE ${into} SET ${setS}`];

            if (n.where) {
                const whereNode = head(n.where);
                const where = recurser(whereNode);
                str.push(`${INDENT}WHERE ${where}${LINE_END}`);
            }
            if (n.limit) {
                const limit = recurser(n.limit);
                str.push(`${INDENT}${limit}`);
            }

            return str.join('');
        },
        transaction : (n) => {
            const isOfActionType = (type) => (action) => (action === type);
            const isBegin = isOfActionType('begin');
            const isRollback = isOfActionType('rollback');

            if (isBegin(n.action)){
                return `${n.action} ${n.defer} TRANSACTION`;
            }
            if (isRollback(n.action)){
                return `ROLLBACK TRANSACTION TO SAVEPOINT ${n.savepoint.name}`;
            }
            return `COMMIT`;
        },
        release : (n) => {
            const recurser = recurse(Generator);
            const savepoint = recurser(n.target.savepoint);
            return `RELEASE SAVEPOINT ${savepoint}`;
        },
        savepoint : (n) => {
            const recurser = recurse(Generator);
            const savepoint = recurser(n.target.savepoint);
            return `SAVEPOINT ${savepoint}`;
        }
    },
    compound : {
        union : (n) => {
            const statement = recurse(Generator)(n.statement);
            return `${toUpper(n.variant)}${LINE_END}${statement}`;
        },
        get 'union all'(){
            return this.union;
        },
        get 'except'(){
            return this.union;
        },
        get 'intersect'(){
            return this.union;
        },
    },
    identifier : {
        star : (n) => n.name,
        table : (n) => {
            const alias =  (n.alias)  ? `AS ${n.alias}` : '';
            const index = (n.index) ? recurse(Generator)(n.index) : '';
            return `${n.name} ${alias} ${index}`;
        },
        index : (n) => `INDEXED BY ${n.name}`,
        column : (n) => {
            const recurser = recurse(Generator);
            const alias =  (n.alias) ? `AS ${n.alias}\`` : '';
            const index = (n.index) ? recurser(n.index) : '';
            return `${n.name} ${alias} ${index}`;
        },
        'function' : (n) => n.name,
        expression : (n) => {
            const m = mapr(Generator);
            return `${n.name}(${m(n.columns)})`;
        },
        view : (n) => n.name,
        savepoint : (n) => n.name,
        trigger : (n) => `"${n.name}"`
    },
    literal : {
        text : (n) => `'${n.value}'`,
        decimal : (n) => `${n.value}`,
        null : () => 'NULL'
    },
    expression : {
        operation : (n) => {
            const recurser = recurse(Generator);
            const isUnaryOperation = isOfFormat('unary');

            if(isUnaryOperation(n)){
                const expression = recurser(n.expression);
                const operator = (n.operator) ? `${n.operator}` : '';
                const alias = (n.alias) ? `AS [${n.alias}]` : '';
                return `${operator} ${expression} ${alias}`;
            }


            const isBetween = (n) => (n.operation === 'between');
            const isExpression = (n) => (n.type === 'expression');

            const side = (s) => {
                const sideOp = recurser(n[s]);
                return !isBetween(n) && (isExpression(n[s]) || containsSelect(sideOp)) ? `(${sideOp})` : sideOp;
            };
            const left = side('left');
            const right = side('right');

            return `${left} ${n.operation} ${right}`;
        },
        list : (n) => {
            const argsList = compose(joinList, mapr(Generator));
            return argsList(n.expression);
        },
        order : (n) => {
            const recurser = recurse(Generator);
            const expression = recurser(n.expression);
            const direction = n.direction;
            return `${expression} ${toUpper(direction)}`;
        },
        limit : (n) => {
            const recurser = recurse(Generator);
            const limit = recurser(n.start);
            const offset = n.offset ? `OFFSET ${recurser(n.offset)}` : '';
            return `LIMIT ${limit}${LINE_END}${INDENT}${offset}`;
        },
        cast : (n) => {
            const recurser = recurse(Generator);
            const expression = recurser(n.expression);
            const as = recurser(n.as);
            const alias = (n.alias) ? `AS [${n.alias}]` : '';
            return `CAST(${expression} AS ${as})${alias}`;
        },
        common : (n) => {
            const recurser = recurse(Generator);
            const expression = recurser(n.expression);
            const target = recurser(n.target);
            return `${target} AS (${expression})`;
        },
        'case' : (n) => {
            const recurser = recurse(Generator);
            const mapConditions = compose(join(LINE_END), mapr(Generator));
            const discriminant = (n.discriminant) ? recurser(n.discriminant) : '';
            const conditions = mapConditions(n.expression);
            const alias = (n.alias) ? `AS [${n.alias}]` : '';
            return `CASE ${discriminant} ${conditions} END ${alias}`;
        },
        recursive : (n) => {
            const recurser = recurse(Generator);
            const target = recurser(n.target);
            const expression = recurser(n.expression);
            return `${target} AS (${expression})`;
        },
        exists : (n) => n.operator
    },
    condition : {
        when : (n) => {
            const recurser = recurse(Generator);
            const when = recurser(n.condition);
            const then = recurser(n.consequent);
            return `WHEN ${when} THEN ${then}`;
        },
        'else' : (n) => {
            const recurser = recurse(Generator);
            const elseS = recurser(n.consequent);
            return `ELSE ${elseS}`;
        },
        'if' : (n) => {
            const recurser = recurse(Generator);
            const exists = recurser(n.condition);
            return `IF ${exists}`;
        }
    },
    'function' : (n) => {
        const recurser = recurse(Generator);
        const name = toUpper(recurser(n.name));
        const args = recurser(n.args);
        const alias =  (n.alias)  ? `AS ${n.alias}\`` : '';
        return `${name}(${args}) ${alias}`;
    },
    module : (n) => {
        const recurser = recurse(Generator);
        const args = recurser(n.args);
        const alias =  (n.alias)  ? `AS ${n.alias}\`` : '';
        return `${n.name}(${args}) ${alias}`;
    },
    event : ({event, occurs, of}) => {
        const processedOf = (of) ? `OF ${mapr(Generator)(of)}` : '';
        return `${occurs} ${event} ${processedOf}`;
    },
    map : {
        join : (n) => {
            const recurser = recurse(Generator);
            const source = recurser(n.source);
            const sourceAlias = (n.source.alias)? n.source.alias : '';
            const join = recurser(head(n.map));

            // Its a select subquery
            if (containsSelect(source)){
                const subquery = `(${source}) AS ${sourceAlias} ${join}`;
                return subquery;
            }
            // Its an inner join.
            return `${source} ${join}`;
        }
    },
    join : {
        join : (n) => {
            const recurser = recurse(Generator);
            const source = recurser(n.source);
            const constraint = recurser(n.constraint);
            return `${INDENT}JOIN ${source}${LINE_END}${constraint}`;
        },
        'inner join' : (n) => {
            const recurser = recurse(Generator);
            const source = recurser(n.source);
            const sourceAlias = (n.source.alias)? ` AS ${n.source.alias}` : '';
            const constraint = recurser(n.constraint);
            return `${INDENT}INNER JOIN (${source})${sourceAlias}${LINE_END}${constraint}`;
        },
        'left outer join' : (n) => {
            const recurser = recurse(Generator);
            const source = recurser(n.source);
            const constraint = recurser(n.constraint);
            return `${INDENT}LEFT OUTER JOIN ${source}${LINE_END}${constraint}`;
        },
        'cross join' : (n) => {
            const recurser = recurse(Generator);
            const source = recurser(n.source);
            return `, ${source}`;
        }
    },
    constraint : {
        join : (n) => {
            const isFormatUsing = isOfFormat('using');
            const isFormatOn = isOfFormat('on');

            const recurser = recurse(Generator);
            if(isFormatOn(n)){
                const on = recurser(n.on);
                return `${INDENT}ON ${on}${LINE_END}`;
            }
            if(isFormatUsing(n)){
                const using = mapr(Generator)(n.using.columns);
                return `${INDENT}USING (${using})${LINE_END}`;
            }
            return '';
        },
        'primary key' : () => `PRIMARY KEY`,
        'not null': () => `NOT NULL`,
        unique : () => `UNIQUE`,
        check : (n) => {
            const check = recurse(Generator)(n.expression);
            return `CHECK (${check})`;
        },
        'foreign key' : (n) => {
            const recurser = recurse(Generator);
            const ref = recurser(n.references);
            return `REFERENCES ${ref}`;
        },
        'null' : () => 'NULL'
    },
    definition : {
        column : (n) => {
            const recurser = recurse(Generator);
            const datatype = isArrayLike(n.datatype) ? mapr(Generator, n.datatype) : recurser(n.datatype);
            const constraintsList = compose(join(' '), map(recurser));
            const constraints = constraintsList(n.definition);
            return `${n.name} ${datatype} ${constraints}`;
        },
        constraint : (n) => {
            const recurser = recurse(Generator);

            const checkConstraint = (type) => (n) => {
                if (isEmpty(n)) { return F;}
                const constraintType = compose(prop('variant'), head);
                return equals(constraintType(n), type);
            };
            const isForeignKey = checkConstraint('foreign key');
            const isPrimaryKey = checkConstraint('primary key');

            if(isForeignKey(n.definition)){
                const childKey = recurser(head(n.columns));
                const parentKey = recurser(head(n.definition));
                return `FOREIGN KEY (${childKey}) ${parentKey}`;
            }
            if(isPrimaryKey(n.definition)){
                const field = recurser(head(n.columns));
                const conflict = prop('conflict', head(n.definition));
                return `PRIMARY KEY (${field}) ON CONFLICT ${conflict}`;
            }
            return recurser(head(n.definition));
        }
    },
    datatype : {
        int : datatype,
        varchar : (n) => {
            const arg = recurse(Generator)(n.args);
            return `${n.variant}(${arg})`;
        },
        blob : datatype,
        double : datatype,
        int8 : datatype,
        text : datatype,
        tinyint : datatype,
        smallint : datatype,
        mediumint :datatype,
        bigint : datatype,
        int4 : datatype,
        integer : datatype,
        time : datatype,
        timestamp : datatype,
        datetime : datatype,
        date : datatype,
        boolean : datatype,
        decimal : (n) => {
            const arg = recurse(Generator)(n.args);
            return `${n.variant}(${arg})`;
        },
        numeric : datatype,
        real : datatype,
        float : datatype,
        'double precision' : datatype,
        clob : (n) => {
            const arg = recurse(Generator)(n.args);
            return `${n.variant}(${arg})`;
        },
        longtext : datatype,
        mediumtext : datatype,
        tinytext : datatype,
        char : (n) => {
            const arg = recurse(Generator)(n.args);
            return `${n.variant}(${arg})`;
        },
        nvarchar : (n) => {
            const arg = recurse(Generator)(n.args);
            return `${n.variant}(${arg})`;
        }

    }
};

module.exports = {
    generate        : (n) => Generator[n.type][n.variant](n)
};