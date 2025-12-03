from flask import Flask, render_template, request, jsonify
import sys
import io

app = Flask(__name__)

# ==========================================
#  COMPILER LOGIC (PHASES 1-6)
# ==========================================

# --- Phase 1: Lexer ---
class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value
    def to_dict(self):
        return {"type": self.type, "value": self.value}

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[0] if self.text else None

    def advance(self):
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def integer(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def identifier(self):
        result = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        return result

    def get_next_token(self):
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            if self.current_char.isdigit():
                return Token('INTEGER', self.integer())
            if self.current_char.isalpha():
                id_str = self.identifier()
                keywords = {'let': 'LET', 'print': 'PRINT', 'if': 'IF', 'while': 'WHILE'}
                return Token(keywords.get(id_str, 'ID'), id_str)

            if self.current_char == '=': self.advance(); return Token('ASSIGN', '=')
            if self.current_char == ';': self.advance(); return Token('SEMI', ';')
            if self.current_char == '+': self.advance(); return Token('PLUS', '+')
            if self.current_char == '-': self.advance(); return Token('MINUS', '-')
            if self.current_char == '*': self.advance(); return Token('MUL', '*')
            if self.current_char == '/': self.advance(); return Token('DIV', '/')
            if self.current_char == '<': self.advance(); return Token('LT', '<')
            if self.current_char == '>': self.advance(); return Token('GT', '>')
            if self.current_char == '{': self.advance(); return Token('LBRACE', '{')
            if self.current_char == '}': self.advance(); return Token('RBRACE', '}')
            if self.current_char == '(': self.advance(); return Token('LPAREN', '(')
            if self.current_char == ')': self.advance(); return Token('RPAREN', ')')

            raise Exception(f'Lexer Error: Invalid character: {self.current_char}')
        return Token('EOF', None)

    def tokenize_all(self):
        tokens = []
        lexer = Lexer(self.text)
        while True:
            tok = lexer.get_next_token()
            tokens.append(tok.to_dict())
            if tok.type == 'EOF': break
        return tokens

# --- Phase 2: Parser ---
class AST:
    def to_dict(self): pass

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left; self.op = op.value; self.right = right
    def to_dict(self): return {"type": "BinOp", "op": self.op, "left": self.left.to_dict(), "right": self.right.to_dict()}

class Num(AST):
    def __init__(self, token):
        self.value = token.value
    def to_dict(self): return {"type": "Num", "value": self.value}

class Var(AST):
    def __init__(self, token):
        self.value = token.value
    def to_dict(self): return {"type": "Var", "value": self.value}

class Assign(AST):
    def __init__(self, left, right):
        self.left = left; self.right = right
    def to_dict(self): return {"type": "Assign", "var": self.left.value, "expr": self.right.to_dict()}

class Print(AST):
    def __init__(self, expr):
        self.expr = expr
    def to_dict(self): return {"type": "Print", "expr": self.expr.to_dict()}

class IfStmt(AST):
    def __init__(self, condition, body):
        self.condition = condition; self.body = body
    def to_dict(self): return {"type": "IfStmt", "condition": self.condition.to_dict(), "body": self.body.to_dict()}

class WhileStmt(AST):
    def __init__(self, condition, body):
        self.condition = condition; self.body = body
    def to_dict(self): return {"type": "WhileStmt", "condition": self.condition.to_dict(), "body": self.body.to_dict()}

class Block(AST):
    def __init__(self, statements):
        self.statements = statements
    def to_dict(self): return {"type": "Block", "stmts": [s.to_dict() for s in self.statements]}

class Parser:
    def __init__(self, lexer):
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def eat(self, token_type):
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            raise Exception(f'Syntax Error: Expected {token_type}, got {self.current_token.type}')

    def factor(self):
        token = self.current_token
        if token.type == 'INTEGER':
            self.eat('INTEGER')
            return Num(token)
        elif token.type == 'ID':
            self.eat('ID')
            return Var(token)
        elif token.type == 'LPAREN':
            self.eat('LPAREN')
            node = self.expr()
            self.eat('RPAREN')
            return node
        raise Exception(f'Parse Error: Unexpected token {token.type}')

    def term(self):
        node = self.factor()
        while self.current_token.type in ('MUL', 'DIV'):
            token = self.current_token
            self.eat(token.type)
            node = BinOp(left=node, op=token, right=self.factor())
        return node

    def expr(self):
        node = self.term()
        while self.current_token.type in ('PLUS', 'MINUS', 'LT', 'GT'):
            token = self.current_token
            self.eat(token.type)
            node = BinOp(left=node, op=token, right=self.term())
        return node

    def block(self):
        self.eat('LBRACE')
        statements = []
        while self.current_token.type != 'RBRACE' and self.current_token.type != 'EOF':
            statements.append(self.statement())
        self.eat('RBRACE')
        return Block(statements)

    def statement(self):
        if self.current_token.type == 'LET':
            self.eat('LET')
            var = Var(self.current_token)
            self.eat('ID')
            self.eat('ASSIGN')
            expr = self.expr()
            self.eat('SEMI')
            return Assign(var, expr)
        elif self.current_token.type == 'ID':
            var = Var(self.current_token)
            self.eat('ID')
            self.eat('ASSIGN')
            expr = self.expr()
            self.eat('SEMI')
            return Assign(var, expr)
        elif self.current_token.type == 'PRINT':
            self.eat('PRINT')
            expr = self.expr()
            self.eat('SEMI')
            return Print(expr)
        elif self.current_token.type == 'IF':
            self.eat('IF')
            self.eat('LPAREN')
            cond = self.expr()
            self.eat('RPAREN')
            body = self.block()
            return IfStmt(cond, body)
        elif self.current_token.type == 'WHILE':
            self.eat('WHILE')
            self.eat('LPAREN')
            cond = self.expr()
            self.eat('RPAREN')
            body = self.block()
            return WhileStmt(cond, body)
        
        raise Exception(f"Unknown statement: {self.current_token.type}")

    def parse(self):
        statements = []
        while self.current_token.type != 'EOF':
            statements.append(self.statement())
        return statements

# --- Phase 3: Semantics ---
class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = set()
    
    def visit(self, node):
        if isinstance(node, Block):
            for stmt in node.statements: self.visit(stmt)
        elif isinstance(node, Assign):
            self.symbol_table.add(node.left.value)
            self.visit(node.right)
        elif isinstance(node, Var):
            if node.value not in self.symbol_table:
                raise Exception(f"Semantic Error: Variable '{node.value}' used before assignment.")
        elif isinstance(node, BinOp):
            self.visit(node.left)
            self.visit(node.right)
        elif isinstance(node, IfStmt) or isinstance(node, WhileStmt):
            self.visit(node.condition)
            self.visit(node.body)
        elif isinstance(node, Print):
            self.visit(node.expr)

    def analyze(self, statements):
        self.symbol_table.clear()
        for stmt in statements:
            self.visit(stmt)
        return list(self.symbol_table)

# --- Phase 4: TAC ---
class TACGenerator:
    def __init__(self):
        self.instructions = []
        self.temp_count = 0
        self.label_count = 0

    def new_temp(self):
        self.temp_count += 1
        return f"t{self.temp_count}"
    
    def new_label(self):
        self.label_count += 1
        return f"L{self.label_count}"

    def gen_expr(self, node):
        if isinstance(node, Num): return str(node.value)
        elif isinstance(node, Var): return node.value
        elif isinstance(node, BinOp):
            l = self.gen_expr(node.left)
            r = self.gen_expr(node.right)
            t = self.new_temp()
            self.instructions.append(f"{t} = {l} {node.op} {r}")
            return t

    def gen_stmt(self, node):
        if isinstance(node, Assign):
            val = self.gen_expr(node.right)
            self.instructions.append(f"{node.left.value} = {val}")
        elif isinstance(node, Print):
            val = self.gen_expr(node.expr)
            self.instructions.append(f"PRINT {val}")
        elif isinstance(node, Block):
            for s in node.statements: self.gen_stmt(s)
        elif isinstance(node, IfStmt):
            cond = self.gen_expr(node.condition)
            end_l = self.new_label()
            self.instructions.append(f"IF_FALSE {cond} GOTO {end_l}")
            self.gen_stmt(node.body)
            self.instructions.append(f"{end_l}:")
        elif isinstance(node, WhileStmt):
            start_l = self.new_label()
            end_l = self.new_label()
            self.instructions.append(f"{start_l}:")
            cond = self.gen_expr(node.condition)
            self.instructions.append(f"IF_FALSE {cond} GOTO {end_l}")
            self.gen_stmt(node.body)
            self.instructions.append(f"GOTO {start_l}")
            self.instructions.append(f"{end_l}:")

    def generate(self, statements):
        self.instructions = []
        for stmt in statements: self.gen_stmt(stmt)
        return self.instructions

# --- Phase 6: VM ---
class VirtualMachine:
    def __init__(self):
        self.memory = {}
        self.output = []

    def resolve(self, val):
        if val.lstrip('-').isdigit(): return int(val)
        return self.memory.get(val, 0)

    def run(self, tac):
        pc = 0
        labels = {line.replace(':', ''): i for i, line in enumerate(tac) if line.endswith(':')}
        steps = 0
        
        while pc < len(tac):
            if steps > 1000: return "Error: Infinite Loop detected"
            steps += 1
            
            line = tac[pc]
            parts = line.split()

            if line.endswith(':'): 
                pc += 1
                continue
            
            if parts[0] == 'PRINT':
                self.output.append(str(self.resolve(parts[1])))
            elif parts[0] == 'GOTO':
                pc = labels[parts[1]]
                continue
            elif parts[0] == 'IF_FALSE':
                cond = self.resolve(parts[1])
                if cond == 0:
                    pc = labels[parts[3]]
                    continue
            elif len(parts) == 3 and parts[1] == '=': # x = 5
                self.memory[parts[0]] = self.resolve(parts[2])
            elif len(parts) == 5: # t0 = a + b
                res, _, v1, op, v2 = parts
                v1 = self.resolve(v1)
                v2 = self.resolve(v2)
                if op == '+': self.memory[res] = v1 + v2
                elif op == '-': self.memory[res] = v1 - v2
                elif op == '*': self.memory[res] = v1 * v2
                elif op == '/': self.memory[res] = v1 // v2
                elif op == '<': self.memory[res] = 1 if v1 < v2 else 0
                elif op == '>': self.memory[res] = 1 if v1 > v2 else 0
            
            pc += 1
        return "\n".join(self.output)

# ==========================================
#  FLASK ROUTES
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/compile', methods=['POST'])
def compile_code():
    data = request.json
    source = data.get('source', '')
    
    response = {
        "tokens": [], "ast": [], "symbols": [], "tac": [], "output": "", "error": None
    }

    try:
        # 1. Lexer
        lexer = Lexer(source)
        response['tokens'] = lexer.tokenize_all()

        # 2. Parser
        lexer_for_parser = Lexer(source)
        parser = Parser(lexer_for_parser)
        ast_nodes = parser.parse()
        response['ast'] = [node.to_dict() for node in ast_nodes]

        # 3. Semantics
        analyzer = SemanticAnalyzer()
        response['symbols'] = analyzer.analyze(ast_nodes)

        # 4. TAC
        tac_gen = TACGenerator()
        tac = tac_gen.generate(ast_nodes)
        response['tac'] = tac

        # 6. VM
        vm = VirtualMachine()
        response['output'] = vm.run(tac)

    except Exception as e:
        response['error'] = str(e)
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)