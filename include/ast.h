#include <stdint.h>
#include <string>

struct TIRExpression {
	enum class ExprType {
		Integer,
		Bool,
		Void,
		Phi,
		Float,
		VariableReference,
		VariableDecl,
		FunctionCall,
		FunctionDefinition,
		Conditional
	};

	ExprType type;

	union {
		intmax_t integer_val;
		bool bool_val;
		double float_val;

		struct {
			std::string name;
		} variable_reference;

		struct {
			std::string name;
			TIRExpression * e1;
			TIRExpression * e2;
		} variable_decl;

		struct {
			TIRExpression * e1;
			TIRExpression * e2;
		} function_call;

		struct {
			std::string arg_name;
			TIRExpression * e1;
		} function_definition;

		struct {
			TIRExpression * condition;
			std::pair<std::string, TIRExpression *> if_block;
			std::pair<std::string, TIRExpression *> else_block;
			TIRExpression * e1;
		} conditional;
	};
};
