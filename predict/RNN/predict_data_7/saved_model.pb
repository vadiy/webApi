??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle???element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements(
handle???element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.8.02v2.8.0-0-g3f878cff5b68??
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:d*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
%simple_rnn_8/simple_rnn_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*6
shared_name'%simple_rnn_8/simple_rnn_cell_8/kernel
?
9simple_rnn_8/simple_rnn_cell_8/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_8/simple_rnn_cell_8/kernel*
_output_shapes

:P*
dtype0
?
/simple_rnn_8/simple_rnn_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*@
shared_name1/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel
?
Csimple_rnn_8/simple_rnn_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel*
_output_shapes

:PP*
dtype0
?
#simple_rnn_8/simple_rnn_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*4
shared_name%#simple_rnn_8/simple_rnn_cell_8/bias
?
7simple_rnn_8/simple_rnn_cell_8/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_8/simple_rnn_cell_8/bias*
_output_shapes
:P*
dtype0
?
%simple_rnn_9/simple_rnn_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*6
shared_name'%simple_rnn_9/simple_rnn_cell_9/kernel
?
9simple_rnn_9/simple_rnn_cell_9/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_9/simple_rnn_cell_9/kernel*
_output_shapes

:Pd*
dtype0
?
/simple_rnn_9/simple_rnn_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*@
shared_name1/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel
?
Csimple_rnn_9/simple_rnn_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel*
_output_shapes

:dd*
dtype0
?
#simple_rnn_9/simple_rnn_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#simple_rnn_9/simple_rnn_cell_9/bias
?
7simple_rnn_9/simple_rnn_cell_9/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_9/simple_rnn_cell_9/bias*
_output_shapes
:d*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0
?
,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*=
shared_name.,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m
?
@Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m*
_output_shapes

:P*
dtype0
?
6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*G
shared_name86Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m
?
JAdam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m*
_output_shapes

:PP*
dtype0
?
*Adam/simple_rnn_8/simple_rnn_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*;
shared_name,*Adam/simple_rnn_8/simple_rnn_cell_8/bias/m
?
>Adam/simple_rnn_8/simple_rnn_cell_8/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_8/simple_rnn_cell_8/bias/m*
_output_shapes
:P*
dtype0
?
,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*=
shared_name.,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m
?
@Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m*
_output_shapes

:Pd*
dtype0
?
6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*G
shared_name86Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m
?
JAdam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m*
_output_shapes

:dd*
dtype0
?
*Adam/simple_rnn_9/simple_rnn_cell_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*;
shared_name,*Adam/simple_rnn_9/simple_rnn_cell_9/bias/m
?
>Adam/simple_rnn_9/simple_rnn_cell_9/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_9/simple_rnn_cell_9/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0
?
,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*=
shared_name.,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v
?
@Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v*
_output_shapes

:P*
dtype0
?
6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*G
shared_name86Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v
?
JAdam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v*
_output_shapes

:PP*
dtype0
?
*Adam/simple_rnn_8/simple_rnn_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*;
shared_name,*Adam/simple_rnn_8/simple_rnn_cell_8/bias/v
?
>Adam/simple_rnn_8/simple_rnn_cell_8/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_8/simple_rnn_cell_8/bias/v*
_output_shapes
:P*
dtype0
?
,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*=
shared_name.,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v
?
@Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v*
_output_shapes

:Pd*
dtype0
?
6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*G
shared_name86Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v
?
JAdam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v*
_output_shapes

:dd*
dtype0
?
*Adam/simple_rnn_9/simple_rnn_cell_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*;
shared_name,*Adam/simple_rnn_9/simple_rnn_cell_9/bias/v
?
>Adam/simple_rnn_9/simple_rnn_cell_9/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_9/simple_rnn_cell_9/bias/v*
_output_shapes
:d*
dtype0

NoOpNoOp
?F
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?F
value?FB?F B?F
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer

signatures
#_self_saveable_object_factories
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature*
?
cell

state_spec
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
* &call_and_return_all_conditional_losses* 
?
!cell
"
state_spec
##_self_saveable_object_factories
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses*
?
#*_self_saveable_object_factories
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses* 
?

2kernel
3bias
#4_self_saveable_object_factories
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses*
?
;iter

<beta_1

=beta_2
	>decay
?learning_rate2m?3m?Am?Bm?Cm?Dm?Em?Fm?2v?3v?Av?Bv?Cv?Dv?Ev?Fv?*

@serving_default* 
* 
<
A0
B1
C2
D3
E4
F5
26
37*
<
A0
B1
C2
D3
E4
F5
26
37*
* 
?
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
?

Akernel
Brecurrent_kernel
Cbias
#L_self_saveable_object_factories
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q_random_generator
R__call__
*S&call_and_return_all_conditional_losses*
* 
* 

A0
B1
C2*

A0
B1
C2*
* 
?

Tstates
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses* 
* 
* 
* 
?

Dkernel
Erecurrent_kernel
Fbias
#__self_saveable_object_factories
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses*
* 
* 

D0
E1
F2*

D0
E1
F2*
* 
?

gstates
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 
* 
* 
* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

20
31*

20
31*
* 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
e_
VARIABLE_VALUE%simple_rnn_8/simple_rnn_cell_8/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_8/simple_rnn_cell_8/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_9/simple_rnn_cell_9/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_9/simple_rnn_cell_9/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

w0*
* 
* 
* 

A0
B1
C2*

A0
B1
C2*
* 
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 

D0
E1
F2*

D0
E1
F2*
* 
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

!0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
?{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_8/simple_rnn_cell_8/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_9/simple_rnn_cell_9/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_8/simple_rnn_cell_8/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE*Adam/simple_rnn_9/simple_rnn_cell_9/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
"serving_default_simple_rnn_8_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_simple_rnn_8_input%simple_rnn_8/simple_rnn_cell_8/kernel#simple_rnn_8/simple_rnn_cell_8/bias/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel%simple_rnn_9/simple_rnn_cell_9/kernel#simple_rnn_9/simple_rnn_cell_9/bias/simple_rnn_9/simple_rnn_cell_9/recurrent_kerneldense_4/kerneldense_4/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_11102811
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9simple_rnn_8/simple_rnn_cell_8/kernel/Read/ReadVariableOpCsimple_rnn_8/simple_rnn_cell_8/recurrent_kernel/Read/ReadVariableOp7simple_rnn_8/simple_rnn_cell_8/bias/Read/ReadVariableOp9simple_rnn_9/simple_rnn_cell_9/kernel/Read/ReadVariableOpCsimple_rnn_9/simple_rnn_cell_9/recurrent_kernel/Read/ReadVariableOp7simple_rnn_9/simple_rnn_cell_9/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp@Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_8/simple_rnn_cell_8/bias/m/Read/ReadVariableOp@Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_9/simple_rnn_cell_9/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp@Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_8/simple_rnn_cell_8/bias/v/Read/ReadVariableOp@Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_9/simple_rnn_cell_9/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_11104076
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%simple_rnn_8/simple_rnn_cell_8/kernel/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel#simple_rnn_8/simple_rnn_cell_8/bias%simple_rnn_9/simple_rnn_cell_9/kernel/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel#simple_rnn_9/simple_rnn_cell_9/biastotalcountAdam/dense_4/kernel/mAdam/dense_4/bias/m,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m*Adam/simple_rnn_8/simple_rnn_cell_8/bias/m,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m*Adam/simple_rnn_9/simple_rnn_cell_9/bias/mAdam/dense_4/kernel/vAdam/dense_4/bias/v,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v6Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v*Adam/simple_rnn_8/simple_rnn_cell_8/bias/v,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v6Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v*Adam/simple_rnn_9/simple_rnn_cell_9/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_11104179??
?=
?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101781

inputsD
2simple_rnn_cell_169_matmul_readvariableop_resource:PdA
3simple_rnn_cell_169_biasadd_readvariableop_resource:dF
4simple_rnn_cell_169_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_169/BiasAdd/ReadVariableOp?)simple_rnn_cell_169/MatMul/ReadVariableOp?+simple_rnn_cell_169/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_169_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_169/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_169_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_169/BiasAddBiasAdd$simple_rnn_cell_169/MatMul:product:02simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_169_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_169/MatMul_1MatMulzeros:output:03simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_169/addAddV2$simple_rnn_cell_169/BiasAdd:output:0&simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_169/TanhTanhsimple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_169_matmul_readvariableop_resource3simple_rnn_cell_169_biasadd_readvariableop_resource4simple_rnn_cell_169_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11101715*
condR
while_cond_11101714*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_169/BiasAdd/ReadVariableOp*^simple_rnn_cell_169/MatMul/ReadVariableOp,^simple_rnn_cell_169/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_169/BiasAdd/ReadVariableOp*simple_rnn_cell_169/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_169/MatMul/ReadVariableOp)simple_rnn_cell_169/MatMul/ReadVariableOp2Z
+simple_rnn_cell_169/MatMul_1/ReadVariableOp+simple_rnn_cell_169/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?9
?
 simple_rnn_9_while_body_111024816
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_25
1simple_rnn_9_while_simple_rnn_9_strided_slice_1_0q
msimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource_0:PdV
Hsimple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource_0:d[
Isimple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0:dd
simple_rnn_9_while_identity!
simple_rnn_9_while_identity_1!
simple_rnn_9_while_identity_2!
simple_rnn_9_while_identity_3!
simple_rnn_9_while_identity_43
/simple_rnn_9_while_simple_rnn_9_strided_slice_1o
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource:PdT
Fsimple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource:dY
Gsimple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource:dd??=simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp?<simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp?>simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp?
Dsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
6simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_9_while_placeholderMsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
<simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
-simple_rnn_9/while/simple_rnn_cell_169/MatMulMatMul=simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
=simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
.simple_rnn_9/while/simple_rnn_cell_169/BiasAddBiasAdd7simple_rnn_9/while/simple_rnn_cell_169/MatMul:product:0Esimple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
/simple_rnn_9/while/simple_rnn_cell_169/MatMul_1MatMul simple_rnn_9_while_placeholder_2Fsimple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_9/while/simple_rnn_cell_169/addAddV27simple_rnn_9/while/simple_rnn_cell_169/BiasAdd:output:09simple_rnn_9/while/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_9/while/simple_rnn_cell_169/TanhTanh.simple_rnn_9/while/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_9_while_placeholder_1simple_rnn_9_while_placeholder/simple_rnn_9/while/simple_rnn_cell_169/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_9/while/addAddV2simple_rnn_9_while_placeholder!simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_9/while/add_1AddV22simple_rnn_9_while_simple_rnn_9_while_loop_counter#simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/add_1:z:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_9/while/Identity_1Identity8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_9/while/Identity_2Identitysimple_rnn_9/while/add:z:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_9/while/Identity_3IdentityGsimple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_9/while/Identity_4Identity/simple_rnn_9/while/simple_rnn_cell_169/Tanh:y:0^simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_9/while/NoOpNoOp>^simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp=^simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp?^simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0"G
simple_rnn_9_while_identity_1&simple_rnn_9/while/Identity_1:output:0"G
simple_rnn_9_while_identity_2&simple_rnn_9/while/Identity_2:output:0"G
simple_rnn_9_while_identity_3&simple_rnn_9/while/Identity_3:output:0"G
simple_rnn_9_while_identity_4&simple_rnn_9/while/Identity_4:output:0"d
/simple_rnn_9_while_simple_rnn_9_strided_slice_11simple_rnn_9_while_simple_rnn_9_strided_slice_1_0"?
Fsimple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resourceHsimple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource_0"?
Gsimple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resourceIsimple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0"?
Esimple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resourceGsimple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource_0"?
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensormsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2~
=simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp=simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp2|
<simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp<simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp2?
>simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp>simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
e
G__inference_dropout_9_layer_call_and_return_conditional_losses_11101794

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
-sequential_4_simple_rnn_8_while_cond_11100782P
Lsequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_while_loop_counterV
Rsequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_while_maximum_iterations/
+sequential_4_simple_rnn_8_while_placeholder1
-sequential_4_simple_rnn_8_while_placeholder_11
-sequential_4_simple_rnn_8_while_placeholder_2R
Nsequential_4_simple_rnn_8_while_less_sequential_4_simple_rnn_8_strided_slice_1j
fsequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_while_cond_11100782___redundant_placeholder0j
fsequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_while_cond_11100782___redundant_placeholder1j
fsequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_while_cond_11100782___redundant_placeholder2j
fsequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_while_cond_11100782___redundant_placeholder3,
(sequential_4_simple_rnn_8_while_identity
?
$sequential_4/simple_rnn_8/while/LessLess+sequential_4_simple_rnn_8_while_placeholderNsequential_4_simple_rnn_8_while_less_sequential_4_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: 
(sequential_4/simple_rnn_8/while/IdentityIdentity(sequential_4/simple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_4_simple_rnn_8_while_identity1sequential_4/simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_11103399
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11103399___redundant_placeholder06
2while_while_cond_11103399___redundant_placeholder16
2while_while_cond_11103399___redundant_placeholder26
2while_while_cond_11103399___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102286
simple_rnn_8_input'
simple_rnn_8_11102264:P#
simple_rnn_8_11102266:P'
simple_rnn_8_11102268:PP'
simple_rnn_9_11102272:Pd#
simple_rnn_9_11102274:d'
simple_rnn_9_11102276:dd"
dense_4_11102280:d
dense_4_11102282:
identity??dense_4/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?$simple_rnn_8/StatefulPartitionedCall?$simple_rnn_9/StatefulPartitionedCall?
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputsimple_rnn_8_11102264simple_rnn_8_11102266simple_rnn_8_11102268*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11102139?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_11102015?
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0simple_rnn_9_11102272simple_rnn_9_11102274simple_rnn_9_11102276*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101986?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_11101862?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_4_11102280dense_4_11102282*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_11101806w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_8_input
?
?
/__inference_simple_rnn_8_layer_call_fn_11102855

inputs
unknown:P
	unknown_0:P
	unknown_1:PP
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11102139s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_11101714
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11101714___redundant_placeholder06
2while_while_cond_11101714___redundant_placeholder16
2while_while_cond_11101714___redundant_placeholder26
2while_while_cond_11101714___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_11101180
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11101180___redundant_placeholder06
2while_while_cond_11101180___redundant_placeholder16
2while_while_cond_11101180___redundant_placeholder26
2while_while_cond_11101180___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11101129

inputs

states0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P2
 matmul_1_readvariableop_resource:PP
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Px
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????PG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????PW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????PY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????P?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????P
 
_user_specified_namestates
?
?
while_cond_11102896
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11102896___redundant_placeholder06
2while_while_cond_11102896___redundant_placeholder16
2while_while_cond_11102896___redundant_placeholder26
2while_while_cond_11102896___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?!
?
while_body_11101314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_169_11101336_0:Pd2
$while_simple_rnn_cell_169_11101338_0:d6
$while_simple_rnn_cell_169_11101340_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_169_11101336:Pd0
"while_simple_rnn_cell_169_11101338:d4
"while_simple_rnn_cell_169_11101340:dd??1while/simple_rnn_cell_169/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_169/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_169_11101336_0$while_simple_rnn_cell_169_11101338_0$while_simple_rnn_cell_169_11101340_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11101301?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_169/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :????
while/Identity_4Identity:while/simple_rnn_cell_169/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_169/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_169_11101336$while_simple_rnn_cell_169_11101336_0"J
"while_simple_rnn_cell_169_11101338$while_simple_rnn_cell_169_11101338_0"J
"while_simple_rnn_cell_169_11101340$while_simple_rnn_cell_169_11101340_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_169/StatefulPartitionedCall1while/simple_rnn_cell_169/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?

?
6__inference_simple_rnn_cell_169_layer_call_fn_11103926

inputs
states_0
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11101421o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
?

?
6__inference_simple_rnn_cell_169_layer_call_fn_11103912

inputs
states_0
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11101301o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????dq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
??
?	
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102554

inputsQ
?simple_rnn_8_simple_rnn_cell_168_matmul_readvariableop_resource:PN
@simple_rnn_8_simple_rnn_cell_168_biasadd_readvariableop_resource:PS
Asimple_rnn_8_simple_rnn_cell_168_matmul_1_readvariableop_resource:PPQ
?simple_rnn_9_simple_rnn_cell_169_matmul_readvariableop_resource:PdN
@simple_rnn_9_simple_rnn_cell_169_biasadd_readvariableop_resource:dS
Asimple_rnn_9_simple_rnn_cell_169_matmul_1_readvariableop_resource:dd8
&dense_4_matmul_readvariableop_resource:d5
'dense_4_biasadd_readvariableop_resource:
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?7simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp?6simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp?8simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp?simple_rnn_8/while?7simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp?6simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp?8simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp?simple_rnn_9/whileH
simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_8/strided_sliceStridedSlicesimple_rnn_8/Shape:output:0)simple_rnn_8/strided_slice/stack:output:0+simple_rnn_8/strided_slice/stack_1:output:0+simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_8/zeros/packedPack#simple_rnn_8/strided_slice:output:0$simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_8/zerosFill"simple_rnn_8/zeros/packed:output:0!simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pp
simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_8/transpose	Transposeinputs$simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:?????????^
simple_rnn_8/Shape_1Shapesimple_rnn_8/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_8/strided_slice_1StridedSlicesimple_rnn_8/Shape_1:output:0+simple_rnn_8/strided_slice_1/stack:output:0-simple_rnn_8/strided_slice_1/stack_1:output:0-simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_8/TensorArrayV2TensorListReserve1simple_rnn_8/TensorArrayV2/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
4simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_8/transpose:y:0Ksimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_8/strided_slice_2StridedSlicesimple_rnn_8/transpose:y:0+simple_rnn_8/strided_slice_2/stack:output:0-simple_rnn_8/strided_slice_2/stack_1:output:0-simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
6simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp?simple_rnn_8_simple_rnn_cell_168_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
'simple_rnn_8/simple_rnn_cell_168/MatMulMatMul%simple_rnn_8/strided_slice_2:output:0>simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_8_simple_rnn_cell_168_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
(simple_rnn_8/simple_rnn_cell_168/BiasAddBiasAdd1simple_rnn_8/simple_rnn_cell_168/MatMul:product:0?simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_8_simple_rnn_cell_168_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
)simple_rnn_8/simple_rnn_cell_168/MatMul_1MatMulsimple_rnn_8/zeros:output:0@simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
$simple_rnn_8/simple_rnn_cell_168/addAddV21simple_rnn_8/simple_rnn_cell_168/BiasAdd:output:03simple_rnn_8/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_8/simple_rnn_cell_168/TanhTanh(simple_rnn_8/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P{
*simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_8/TensorArrayV2_1TensorListReserve3simple_rnn_8/TensorArrayV2_1/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_8/whileWhile(simple_rnn_8/while/loop_counter:output:0.simple_rnn_8/while/maximum_iterations:output:0simple_rnn_8/time:output:0%simple_rnn_8/TensorArrayV2_1:handle:0simple_rnn_8/zeros:output:0%simple_rnn_8/strided_slice_1:output:0Dsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_8_simple_rnn_cell_168_matmul_readvariableop_resource@simple_rnn_8_simple_rnn_cell_168_biasadd_readvariableop_resourceAsimple_rnn_8_simple_rnn_cell_168_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_8_while_body_11102376*,
cond$R"
 simple_rnn_8_while_cond_11102375*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
=simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
/simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_8/while:output:3Fsimple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0u
"simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_8/strided_slice_3StridedSlice8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_8/strided_slice_3/stack:output:0-simple_rnn_8/strided_slice_3/stack_1:output:0-simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maskr
simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_8/transpose_1	Transpose8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pr
dropout_8/IdentityIdentitysimple_rnn_8/transpose_1:y:0*
T0*+
_output_shapes
:?????????P]
simple_rnn_9/ShapeShapedropout_8/Identity:output:0*
T0*
_output_shapes
:j
 simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_9/strided_sliceStridedSlicesimple_rnn_9/Shape:output:0)simple_rnn_9/strided_slice/stack:output:0+simple_rnn_9/strided_slice/stack_1:output:0+simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_9/zeros/packedPack#simple_rnn_9/strided_slice:output:0$simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_9/zerosFill"simple_rnn_9/zeros/packed:output:0!simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dp
simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_9/transpose	Transposedropout_8/Identity:output:0$simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P^
simple_rnn_9/Shape_1Shapesimple_rnn_9/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_9/strided_slice_1StridedSlicesimple_rnn_9/Shape_1:output:0+simple_rnn_9/strided_slice_1/stack:output:0-simple_rnn_9/strided_slice_1/stack_1:output:0-simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_9/TensorArrayV2TensorListReserve1simple_rnn_9/TensorArrayV2/element_shape:output:0%simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
4simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_9/transpose:y:0Ksimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_9/strided_slice_2StridedSlicesimple_rnn_9/transpose:y:0+simple_rnn_9/strided_slice_2/stack:output:0-simple_rnn_9/strided_slice_2/stack_1:output:0-simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
6simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp?simple_rnn_9_simple_rnn_cell_169_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
'simple_rnn_9/simple_rnn_cell_169/MatMulMatMul%simple_rnn_9/strided_slice_2:output:0>simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_9_simple_rnn_cell_169_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
(simple_rnn_9/simple_rnn_cell_169/BiasAddBiasAdd1simple_rnn_9/simple_rnn_cell_169/MatMul:product:0?simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_9_simple_rnn_cell_169_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
)simple_rnn_9/simple_rnn_cell_169/MatMul_1MatMulsimple_rnn_9/zeros:output:0@simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
$simple_rnn_9/simple_rnn_cell_169/addAddV21simple_rnn_9/simple_rnn_cell_169/BiasAdd:output:03simple_rnn_9/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_9/simple_rnn_cell_169/TanhTanh(simple_rnn_9/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d{
*simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_9/TensorArrayV2_1TensorListReserve3simple_rnn_9/TensorArrayV2_1/element_shape:output:0%simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_9/whileWhile(simple_rnn_9/while/loop_counter:output:0.simple_rnn_9/while/maximum_iterations:output:0simple_rnn_9/time:output:0%simple_rnn_9/TensorArrayV2_1:handle:0simple_rnn_9/zeros:output:0%simple_rnn_9/strided_slice_1:output:0Dsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_9_simple_rnn_cell_169_matmul_readvariableop_resource@simple_rnn_9_simple_rnn_cell_169_biasadd_readvariableop_resourceAsimple_rnn_9_simple_rnn_cell_169_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_9_while_body_11102481*,
cond$R"
 simple_rnn_9_while_cond_11102480*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
=simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
/simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_9/while:output:3Fsimple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0u
"simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_9/strided_slice_3StridedSlice8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_9/strided_slice_3/stack:output:0-simple_rnn_9/strided_slice_3/stack_1:output:0-simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maskr
simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_9/transpose_1	Transpose8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dw
dropout_9/IdentityIdentity%simple_rnn_9/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_4/MatMulMatMuldropout_9/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp8^simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp7^simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp9^simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp^simple_rnn_8/while8^simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp7^simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp9^simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp^simple_rnn_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2r
7simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp7simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp2p
6simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp6simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp2t
8simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp8simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp2(
simple_rnn_8/whilesimple_rnn_8/while2r
7simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp7simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp2p
6simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp6simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp2t
8simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp8simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp2(
simple_rnn_9/whilesimple_rnn_9/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_simple_rnn_9_layer_call_fn_11103358

inputs
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101986o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?>
?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103574
inputs_0D
2simple_rnn_cell_169_matmul_readvariableop_resource:PdA
3simple_rnn_cell_169_biasadd_readvariableop_resource:dF
4simple_rnn_cell_169_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_169/BiasAdd/ReadVariableOp?)simple_rnn_cell_169/MatMul/ReadVariableOp?+simple_rnn_cell_169/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_169_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_169/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_169_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_169/BiasAddBiasAdd$simple_rnn_cell_169/MatMul:product:02simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_169_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_169/MatMul_1MatMulzeros:output:03simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_169/addAddV2$simple_rnn_cell_169/BiasAdd:output:0&simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_169/TanhTanhsimple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_169_matmul_readvariableop_resource3simple_rnn_cell_169_biasadd_readvariableop_resource4simple_rnn_cell_169_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11103508*
condR
while_cond_11103507*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_169/BiasAdd/ReadVariableOp*^simple_rnn_cell_169/MatMul/ReadVariableOp,^simple_rnn_cell_169/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_169/BiasAdd/ReadVariableOp*simple_rnn_cell_169/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_169/MatMul/ReadVariableOp)simple_rnn_cell_169/MatMul/ReadVariableOp2Z
+simple_rnn_cell_169/MatMul_1/ReadVariableOp+simple_rnn_cell_169/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?
?
while_cond_11103112
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11103112___redundant_placeholder06
2while_while_cond_11103112___redundant_placeholder16
2while_while_cond_11103112___redundant_placeholder26
2while_while_cond_11103112___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
/__inference_simple_rnn_8_layer_call_fn_11102822
inputs_0
unknown:P
	unknown_0:P
	unknown_1:PP
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11101085|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?-
?
while_body_11103724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_169_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_169_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_169_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_169/MatMul/ReadVariableOp?1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_169_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_169/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_169/BiasAddBiasAdd*while/simple_rnn_cell_169/MatMul:product:08while/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_169/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_169/addAddV2*while/simple_rnn_cell_169/BiasAdd:output:0,while/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_169/TanhTanh!while/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_169/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_169/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_169/MatMul/ReadVariableOp2^while/simple_rnn_cell_169/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_169_biasadd_readvariableop_resource;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_169_matmul_readvariableop_resource:while_simple_rnn_cell_169_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_169/MatMul/ReadVariableOp/while/simple_rnn_cell_169/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11101421

inputs

states0
matmul_readvariableop_resource:Pd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????dG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates
?
?
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11103881

inputs
states_00
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P2
 matmul_1_readvariableop_resource:PP
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Px
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????PG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????PW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????PY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????P?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????P
"
_user_specified_name
states/0
?9
?
 simple_rnn_9_while_body_111027086
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_25
1simple_rnn_9_while_simple_rnn_9_strided_slice_1_0q
msimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource_0:PdV
Hsimple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource_0:d[
Isimple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0:dd
simple_rnn_9_while_identity!
simple_rnn_9_while_identity_1!
simple_rnn_9_while_identity_2!
simple_rnn_9_while_identity_3!
simple_rnn_9_while_identity_43
/simple_rnn_9_while_simple_rnn_9_strided_slice_1o
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource:PdT
Fsimple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource:dY
Gsimple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource:dd??=simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp?<simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp?>simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp?
Dsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
6simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_9_while_placeholderMsimple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
<simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
-simple_rnn_9/while/simple_rnn_cell_169/MatMulMatMul=simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
=simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
.simple_rnn_9/while/simple_rnn_cell_169/BiasAddBiasAdd7simple_rnn_9/while/simple_rnn_cell_169/MatMul:product:0Esimple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
/simple_rnn_9/while/simple_rnn_cell_169/MatMul_1MatMul simple_rnn_9_while_placeholder_2Fsimple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_9/while/simple_rnn_cell_169/addAddV27simple_rnn_9/while/simple_rnn_cell_169/BiasAdd:output:09simple_rnn_9/while/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_9/while/simple_rnn_cell_169/TanhTanh.simple_rnn_9/while/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_9_while_placeholder_1simple_rnn_9_while_placeholder/simple_rnn_9/while/simple_rnn_cell_169/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_9/while/addAddV2simple_rnn_9_while_placeholder!simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_9/while/add_1AddV22simple_rnn_9_while_simple_rnn_9_while_loop_counter#simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/add_1:z:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_9/while/Identity_1Identity8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_9/while/Identity_2Identitysimple_rnn_9/while/add:z:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_9/while/Identity_3IdentityGsimple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_9/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_9/while/Identity_4Identity/simple_rnn_9/while/simple_rnn_cell_169/Tanh:y:0^simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_9/while/NoOpNoOp>^simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp=^simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp?^simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0"G
simple_rnn_9_while_identity_1&simple_rnn_9/while/Identity_1:output:0"G
simple_rnn_9_while_identity_2&simple_rnn_9/while/Identity_2:output:0"G
simple_rnn_9_while_identity_3&simple_rnn_9/while/Identity_3:output:0"G
simple_rnn_9_while_identity_4&simple_rnn_9/while/Identity_4:output:0"d
/simple_rnn_9_while_simple_rnn_9_strided_slice_11simple_rnn_9_while_simple_rnn_9_strided_slice_1_0"?
Fsimple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resourceHsimple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource_0"?
Gsimple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resourceIsimple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0"?
Esimple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resourceGsimple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource_0"?
ksimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensormsimple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2~
=simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp=simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp2|
<simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp<simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp2?
>simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp>simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?=
?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11102139

inputsD
2simple_rnn_cell_168_matmul_readvariableop_resource:PA
3simple_rnn_cell_168_biasadd_readvariableop_resource:PF
4simple_rnn_cell_168_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_168/BiasAdd/ReadVariableOp?)simple_rnn_cell_168/MatMul/ReadVariableOp?+simple_rnn_cell_168/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_168_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_168/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_168_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_168/BiasAddBiasAdd$simple_rnn_cell_168/MatMul:product:02simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_168_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_168/MatMul_1MatMulzeros:output:03simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_168/addAddV2$simple_rnn_cell_168/BiasAdd:output:0&simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_168/TanhTanhsimple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_168_matmul_readvariableop_resource3simple_rnn_cell_168_biasadd_readvariableop_resource4simple_rnn_cell_168_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11102073*
condR
while_cond_11102072*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_168/BiasAdd/ReadVariableOp*^simple_rnn_cell_168/MatMul/ReadVariableOp,^simple_rnn_cell_168/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_168/BiasAdd/ReadVariableOp*simple_rnn_cell_168/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_168/MatMul/ReadVariableOp)simple_rnn_cell_168/MatMul/ReadVariableOp2Z
+simple_rnn_cell_168/MatMul_1/ReadVariableOp+simple_rnn_cell_168/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101536

inputs.
simple_rnn_cell_169_11101461:Pd*
simple_rnn_cell_169_11101463:d.
simple_rnn_cell_169_11101465:dd
identity??+simple_rnn_cell_169/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
+simple_rnn_cell_169/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_169_11101461simple_rnn_cell_169_11101463simple_rnn_cell_169_11101465*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11101421n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_169_11101461simple_rnn_cell_169_11101463simple_rnn_cell_169_11101465*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11101473*
condR
while_cond_11101472*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d|
NoOpNoOp,^simple_rnn_cell_169/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_169/StatefulPartitionedCall+simple_rnn_cell_169/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?
e
,__inference_dropout_8_layer_call_fn_11103297

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_11102015s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?=
?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11103287

inputsD
2simple_rnn_cell_168_matmul_readvariableop_resource:PA
3simple_rnn_cell_168_biasadd_readvariableop_resource:PF
4simple_rnn_cell_168_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_168/BiasAdd/ReadVariableOp?)simple_rnn_cell_168/MatMul/ReadVariableOp?+simple_rnn_cell_168/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_168_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_168/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_168_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_168/BiasAddBiasAdd$simple_rnn_cell_168/MatMul:product:02simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_168_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_168/MatMul_1MatMulzeros:output:03simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_168/addAddV2$simple_rnn_cell_168/BiasAdd:output:0&simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_168/TanhTanhsimple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_168_matmul_readvariableop_resource3simple_rnn_cell_168_biasadd_readvariableop_resource4simple_rnn_cell_168_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11103221*
condR
while_cond_11103220*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_168/BiasAdd/ReadVariableOp*^simple_rnn_cell_168/MatMul/ReadVariableOp,^simple_rnn_cell_168/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_168/BiasAdd/ReadVariableOp*simple_rnn_cell_168/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_168/MatMul/ReadVariableOp)simple_rnn_cell_168/MatMul/ReadVariableOp2Z
+simple_rnn_cell_168/MatMul_1/ReadVariableOp+simple_rnn_cell_168/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11103960

inputs
states_00
matmul_readvariableop_resource:Pd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????dG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
?

?
 simple_rnn_9_while_cond_111024806
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_28
4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1P
Lsimple_rnn_9_while_simple_rnn_9_while_cond_11102480___redundant_placeholder0P
Lsimple_rnn_9_while_simple_rnn_9_while_cond_11102480___redundant_placeholder1P
Lsimple_rnn_9_while_simple_rnn_9_while_cond_11102480___redundant_placeholder2P
Lsimple_rnn_9_while_simple_rnn_9_while_cond_11102480___redundant_placeholder3
simple_rnn_9_while_identity
?
simple_rnn_9/while/LessLesssimple_rnn_9_while_placeholder4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?-
?
while_body_11101715
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_169_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_169_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_169_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_169/MatMul/ReadVariableOp?1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_169_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_169/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_169/BiasAddBiasAdd*while/simple_rnn_cell_169/MatMul:product:08while/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_169/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_169/addAddV2*while/simple_rnn_cell_169/BiasAdd:output:0,while/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_169/TanhTanh!while/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_169/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_169/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_169/MatMul/ReadVariableOp2^while/simple_rnn_cell_169/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_169_biasadd_readvariableop_resource;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_169_matmul_readvariableop_resource:while_simple_rnn_cell_169_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_169/MatMul/ReadVariableOp/while/simple_rnn_cell_169/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?!
?
while_body_11101181
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_168_11101203_0:P2
$while_simple_rnn_cell_168_11101205_0:P6
$while_simple_rnn_cell_168_11101207_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_168_11101203:P0
"while_simple_rnn_cell_168_11101205:P4
"while_simple_rnn_cell_168_11101207:PP??1while/simple_rnn_cell_168/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_168/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_168_11101203_0$while_simple_rnn_cell_168_11101205_0$while_simple_rnn_cell_168_11101207_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11101129?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_168/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :????
while/Identity_4Identity:while/simple_rnn_cell_168/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_168/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_168_11101203$while_simple_rnn_cell_168_11101203_0"J
"while_simple_rnn_cell_168_11101205$while_simple_rnn_cell_168_11101205_0"J
"while_simple_rnn_cell_168_11101207$while_simple_rnn_cell_168_11101207_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_168/StatefulPartitionedCall1while/simple_rnn_cell_168/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_11101472
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11101472___redundant_placeholder06
2while_while_cond_11101472___redundant_placeholder16
2while_while_cond_11101472___redundant_placeholder26
2while_while_cond_11101472___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?-
?
while_body_11103221
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_168_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_168_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_168_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_168/MatMul/ReadVariableOp?1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_168_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_168/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_168/BiasAddBiasAdd*while/simple_rnn_cell_168/MatMul:product:08while/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_168/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_168/addAddV2*while/simple_rnn_cell_168/BiasAdd:output:0,while/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_168/TanhTanh!while/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_168/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_168/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_168/MatMul/ReadVariableOp2^while/simple_rnn_cell_168/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_168_biasadd_readvariableop_resource;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_168_matmul_readvariableop_resource:while_simple_rnn_cell_168_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_168/MatMul/ReadVariableOp/while/simple_rnn_cell_168/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?=
?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103682

inputsD
2simple_rnn_cell_169_matmul_readvariableop_resource:PdA
3simple_rnn_cell_169_biasadd_readvariableop_resource:dF
4simple_rnn_cell_169_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_169/BiasAdd/ReadVariableOp?)simple_rnn_cell_169/MatMul/ReadVariableOp?+simple_rnn_cell_169/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_169_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_169/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_169_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_169/BiasAddBiasAdd$simple_rnn_cell_169/MatMul:product:02simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_169_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_169/MatMul_1MatMulzeros:output:03simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_169/addAddV2$simple_rnn_cell_169/BiasAdd:output:0&simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_169/TanhTanhsimple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_169_matmul_readvariableop_resource3simple_rnn_cell_169_biasadd_readvariableop_resource4simple_rnn_cell_169_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11103616*
condR
while_cond_11103615*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_169/BiasAdd/ReadVariableOp*^simple_rnn_cell_169/MatMul/ReadVariableOp,^simple_rnn_cell_169/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_169/BiasAdd/ReadVariableOp*simple_rnn_cell_169/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_169/MatMul/ReadVariableOp)simple_rnn_cell_169/MatMul/ReadVariableOp2Z
+simple_rnn_cell_169/MatMul_1/ReadVariableOp+simple_rnn_cell_169/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
/__inference_simple_rnn_9_layer_call_fn_11103347

inputs
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101781o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
e
G__inference_dropout_8_layer_call_and_return_conditional_losses_11101672

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_11101813

inputs'
simple_rnn_8_11101660:P#
simple_rnn_8_11101662:P'
simple_rnn_8_11101664:PP'
simple_rnn_9_11101782:Pd#
simple_rnn_9_11101784:d'
simple_rnn_9_11101786:dd"
dense_4_11101807:d
dense_4_11101809:
identity??dense_4/StatefulPartitionedCall?$simple_rnn_8/StatefulPartitionedCall?$simple_rnn_9/StatefulPartitionedCall?
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_8_11101660simple_rnn_8_11101662simple_rnn_8_11101664*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11101659?
dropout_8/PartitionedCallPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_11101672?
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0simple_rnn_9_11101782simple_rnn_9_11101784simple_rnn_9_11101786*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101781?
dropout_9/PartitionedCallPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_11101794?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_4_11101807dense_4_11101809*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_11101806w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102196

inputs'
simple_rnn_8_11102174:P#
simple_rnn_8_11102176:P'
simple_rnn_8_11102178:PP'
simple_rnn_9_11102182:Pd#
simple_rnn_9_11102184:d'
simple_rnn_9_11102186:dd"
dense_4_11102190:d
dense_4_11102192:
identity??dense_4/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?!dropout_9/StatefulPartitionedCall?$simple_rnn_8/StatefulPartitionedCall?$simple_rnn_9/StatefulPartitionedCall?
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_8_11102174simple_rnn_8_11102176simple_rnn_8_11102178*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11102139?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_11102015?
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0simple_rnn_9_11102182simple_rnn_9_11102184simple_rnn_9_11102186*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101986?
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_11101862?
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_4_11102190dense_4_11102192*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_11101806w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11101085

inputs.
simple_rnn_cell_168_11101010:P*
simple_rnn_cell_168_11101012:P.
simple_rnn_cell_168_11101014:PP
identity??+simple_rnn_cell_168/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
+simple_rnn_cell_168/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_168_11101010simple_rnn_cell_168_11101012simple_rnn_cell_168_11101014*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11101009n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_168_11101010simple_rnn_cell_168_11101012simple_rnn_cell_168_11101014*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11101022*
condR
while_cond_11101021*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????Pk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P|
NoOpNoOp,^simple_rnn_cell_168/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_168/StatefulPartitionedCall+simple_rnn_cell_168/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?-
?
while_body_11103400
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_169_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_169_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_169_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_169/MatMul/ReadVariableOp?1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_169_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_169/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_169/BiasAddBiasAdd*while/simple_rnn_cell_169/MatMul:product:08while/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_169/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_169/addAddV2*while/simple_rnn_cell_169/BiasAdd:output:0,while/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_169/TanhTanh!while/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_169/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_169/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_169/MatMul/ReadVariableOp2^while/simple_rnn_cell_169/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_169_biasadd_readvariableop_resource;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_169_matmul_readvariableop_resource:while_simple_rnn_cell_169_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_169/MatMul/ReadVariableOp/while/simple_rnn_cell_169/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?-
?
while_body_11102897
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_168_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_168_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_168_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_168/MatMul/ReadVariableOp?1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_168_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_168/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_168/BiasAddBiasAdd*while/simple_rnn_cell_168/MatMul:product:08while/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_168/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_168/addAddV2*while/simple_rnn_cell_168/BiasAdd:output:0,while/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_168/TanhTanh!while/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_168/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_168/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_168/MatMul/ReadVariableOp2^while/simple_rnn_cell_168/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_168_biasadd_readvariableop_resource;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_168_matmul_readvariableop_resource:while_simple_rnn_cell_168_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_168/MatMul/ReadVariableOp/while/simple_rnn_cell_168/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_simple_rnn_8_layer_call_fn_11102844

inputs
unknown:P
	unknown_0:P
	unknown_1:PP
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11101659s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_11101021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11101021___redundant_placeholder06
2while_while_cond_11101021___redundant_placeholder16
2while_while_cond_11101021___redundant_placeholder26
2while_while_cond_11101021___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?>
?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103466
inputs_0D
2simple_rnn_cell_169_matmul_readvariableop_resource:PdA
3simple_rnn_cell_169_biasadd_readvariableop_resource:dF
4simple_rnn_cell_169_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_169/BiasAdd/ReadVariableOp?)simple_rnn_cell_169/MatMul/ReadVariableOp?+simple_rnn_cell_169/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_169_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_169/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_169_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_169/BiasAddBiasAdd$simple_rnn_cell_169/MatMul:product:02simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_169_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_169/MatMul_1MatMulzeros:output:03simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_169/addAddV2$simple_rnn_cell_169/BiasAdd:output:0&simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_169/TanhTanhsimple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_169_matmul_readvariableop_resource3simple_rnn_cell_169_biasadd_readvariableop_resource4simple_rnn_cell_169_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11103400*
condR
while_cond_11103399*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_169/BiasAdd/ReadVariableOp*^simple_rnn_cell_169/MatMul/ReadVariableOp,^simple_rnn_cell_169/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_169/BiasAdd/ReadVariableOp*simple_rnn_cell_169/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_169/MatMul/ReadVariableOp)simple_rnn_cell_169/MatMul/ReadVariableOp2Z
+simple_rnn_cell_169/MatMul_1/ReadVariableOp+simple_rnn_cell_169/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?-
?
while_body_11103113
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_168_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_168_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_168_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_168/MatMul/ReadVariableOp?1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_168_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_168/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_168/BiasAddBiasAdd*while/simple_rnn_cell_168/MatMul:product:08while/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_168/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_168/addAddV2*while/simple_rnn_cell_168/BiasAdd:output:0,while/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_168/TanhTanh!while/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_168/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_168/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_168/MatMul/ReadVariableOp2^while/simple_rnn_cell_168/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_168_biasadd_readvariableop_resource;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_168_matmul_readvariableop_resource:while_simple_rnn_cell_168_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_168/MatMul/ReadVariableOp/while/simple_rnn_cell_168/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?

?
 simple_rnn_8_while_cond_111023756
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_28
4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1P
Lsimple_rnn_8_while_simple_rnn_8_while_cond_11102375___redundant_placeholder0P
Lsimple_rnn_8_while_simple_rnn_8_while_cond_11102375___redundant_placeholder1P
Lsimple_rnn_8_while_simple_rnn_8_while_cond_11102375___redundant_placeholder2P
Lsimple_rnn_8_while_simple_rnn_8_while_cond_11102375___redundant_placeholder3
simple_rnn_8_while_identity
?
simple_rnn_8/while/LessLesssimple_rnn_8_while_placeholder4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?!
?
while_body_11101022
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_168_11101044_0:P2
$while_simple_rnn_cell_168_11101046_0:P6
$while_simple_rnn_cell_168_11101048_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_168_11101044:P0
"while_simple_rnn_cell_168_11101046:P4
"while_simple_rnn_cell_168_11101048:PP??1while/simple_rnn_cell_168/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_168/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_168_11101044_0$while_simple_rnn_cell_168_11101046_0$while_simple_rnn_cell_168_11101048_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11101009?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_168/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :????
while/Identity_4Identity:while/simple_rnn_cell_168/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_168/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_168_11101044$while_simple_rnn_cell_168_11101044_0"J
"while_simple_rnn_cell_168_11101046$while_simple_rnn_cell_168_11101046_0"J
"while_simple_rnn_cell_168_11101048$while_simple_rnn_cell_168_11101048_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_168/StatefulPartitionedCall1while/simple_rnn_cell_168/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_11103220
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11103220___redundant_placeholder06
2while_while_cond_11103220___redundant_placeholder16
2while_while_cond_11103220___redundant_placeholder26
2while_while_cond_11103220___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?

f
G__inference_dropout_8_layer_call_and_return_conditional_losses_11102015

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	
?
/__inference_sequential_4_layer_call_fn_11102334

inputs
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102196o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11103943

inputs
states_00
matmul_readvariableop_resource:Pd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????dG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????d
"
_user_specified_name
states/0
?-
?
while_body_11103616
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_169_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_169_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_169_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_169/MatMul/ReadVariableOp?1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_169_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_169/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_169/BiasAddBiasAdd*while/simple_rnn_cell_169/MatMul:product:08while/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_169/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_169/addAddV2*while/simple_rnn_cell_169/BiasAdd:output:0,while/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_169/TanhTanh!while/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_169/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_169/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_169/MatMul/ReadVariableOp2^while/simple_rnn_cell_169/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_169_biasadd_readvariableop_resource;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_169_matmul_readvariableop_resource:while_simple_rnn_cell_169_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_169/MatMul/ReadVariableOp/while/simple_rnn_cell_169/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?=
?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11101659

inputsD
2simple_rnn_cell_168_matmul_readvariableop_resource:PA
3simple_rnn_cell_168_biasadd_readvariableop_resource:PF
4simple_rnn_cell_168_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_168/BiasAdd/ReadVariableOp?)simple_rnn_cell_168/MatMul/ReadVariableOp?+simple_rnn_cell_168/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_168_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_168/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_168_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_168/BiasAddBiasAdd$simple_rnn_cell_168/MatMul:product:02simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_168_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_168/MatMul_1MatMulzeros:output:03simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_168/addAddV2$simple_rnn_cell_168/BiasAdd:output:0&simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_168/TanhTanhsimple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_168_matmul_readvariableop_resource3simple_rnn_cell_168_biasadd_readvariableop_resource4simple_rnn_cell_168_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11101593*
condR
while_cond_11101592*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_168/BiasAdd/ReadVariableOp*^simple_rnn_cell_168/MatMul/ReadVariableOp,^simple_rnn_cell_168/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_168/BiasAdd/ReadVariableOp*simple_rnn_cell_168/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_168/MatMul/ReadVariableOp)simple_rnn_cell_168/MatMul/ReadVariableOp2Z
+simple_rnn_cell_168/MatMul_1/ReadVariableOp+simple_rnn_cell_168/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
while_body_11102073
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_168_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_168_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_168_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_168/MatMul/ReadVariableOp?1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_168_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_168/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_168/BiasAddBiasAdd*while/simple_rnn_cell_168/MatMul:product:08while/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_168/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_168/addAddV2*while/simple_rnn_cell_168/BiasAdd:output:0,while/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_168/TanhTanh!while/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_168/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_168/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_168/MatMul/ReadVariableOp2^while/simple_rnn_cell_168/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_168_biasadd_readvariableop_resource;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_168_matmul_readvariableop_resource:while_simple_rnn_cell_168_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_168/MatMul/ReadVariableOp/while/simple_rnn_cell_168/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_11101919
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11101919___redundant_placeholder06
2while_while_cond_11101919___redundant_placeholder16
2while_while_cond_11101919___redundant_placeholder26
2while_while_cond_11101919___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?	
?
/__inference_sequential_4_layer_call_fn_11102236
simple_rnn_8_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102196o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_8_input
?
H
,__inference_dropout_8_layer_call_fn_11103292

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_11101672d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?

?
6__inference_simple_rnn_cell_168_layer_call_fn_11103850

inputs
states_0
unknown:P
	unknown_0:P
	unknown_1:PP
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11101009o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Pq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????P
"
_user_specified_name
states/0
ҵ
?

#__inference__wrapped_model_11100961
simple_rnn_8_input^
Lsequential_4_simple_rnn_8_simple_rnn_cell_168_matmul_readvariableop_resource:P[
Msequential_4_simple_rnn_8_simple_rnn_cell_168_biasadd_readvariableop_resource:P`
Nsequential_4_simple_rnn_8_simple_rnn_cell_168_matmul_1_readvariableop_resource:PP^
Lsequential_4_simple_rnn_9_simple_rnn_cell_169_matmul_readvariableop_resource:Pd[
Msequential_4_simple_rnn_9_simple_rnn_cell_169_biasadd_readvariableop_resource:d`
Nsequential_4_simple_rnn_9_simple_rnn_cell_169_matmul_1_readvariableop_resource:ddE
3sequential_4_dense_4_matmul_readvariableop_resource:dB
4sequential_4_dense_4_biasadd_readvariableop_resource:
identity??+sequential_4/dense_4/BiasAdd/ReadVariableOp?*sequential_4/dense_4/MatMul/ReadVariableOp?Dsequential_4/simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp?Csequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp?Esequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp?sequential_4/simple_rnn_8/while?Dsequential_4/simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp?Csequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp?Esequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp?sequential_4/simple_rnn_9/whilea
sequential_4/simple_rnn_8/ShapeShapesimple_rnn_8_input*
T0*
_output_shapes
:w
-sequential_4/simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_4/simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_4/simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'sequential_4/simple_rnn_8/strided_sliceStridedSlice(sequential_4/simple_rnn_8/Shape:output:06sequential_4/simple_rnn_8/strided_slice/stack:output:08sequential_4/simple_rnn_8/strided_slice/stack_1:output:08sequential_4/simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_4/simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
&sequential_4/simple_rnn_8/zeros/packedPack0sequential_4/simple_rnn_8/strided_slice:output:01sequential_4/simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_4/simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_4/simple_rnn_8/zerosFill/sequential_4/simple_rnn_8/zeros/packed:output:0.sequential_4/simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:?????????P}
(sequential_4/simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
#sequential_4/simple_rnn_8/transpose	Transposesimple_rnn_8_input1sequential_4/simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:?????????x
!sequential_4/simple_rnn_8/Shape_1Shape'sequential_4/simple_rnn_8/transpose:y:0*
T0*
_output_shapes
:y
/sequential_4/simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_4/simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_4/simple_rnn_8/strided_slice_1StridedSlice*sequential_4/simple_rnn_8/Shape_1:output:08sequential_4/simple_rnn_8/strided_slice_1/stack:output:0:sequential_4/simple_rnn_8/strided_slice_1/stack_1:output:0:sequential_4/simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_4/simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_4/simple_rnn_8/TensorArrayV2TensorListReserve>sequential_4/simple_rnn_8/TensorArrayV2/element_shape:output:02sequential_4/simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Osequential_4/simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Asequential_4/simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_4/simple_rnn_8/transpose:y:0Xsequential_4/simple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???y
/sequential_4/simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_4/simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_4/simple_rnn_8/strided_slice_2StridedSlice'sequential_4/simple_rnn_8/transpose:y:08sequential_4/simple_rnn_8/strided_slice_2/stack:output:0:sequential_4/simple_rnn_8/strided_slice_2/stack_1:output:0:sequential_4/simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
Csequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOpLsequential_4_simple_rnn_8_simple_rnn_cell_168_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
4sequential_4/simple_rnn_8/simple_rnn_cell_168/MatMulMatMul2sequential_4/simple_rnn_8/strided_slice_2:output:0Ksequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Dsequential_4/simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOpMsequential_4_simple_rnn_8_simple_rnn_cell_168_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
5sequential_4/simple_rnn_8/simple_rnn_cell_168/BiasAddBiasAdd>sequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul:product:0Lsequential_4/simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Esequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOpNsequential_4_simple_rnn_8_simple_rnn_cell_168_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
6sequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul_1MatMul(sequential_4/simple_rnn_8/zeros:output:0Msequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1sequential_4/simple_rnn_8/simple_rnn_cell_168/addAddV2>sequential_4/simple_rnn_8/simple_rnn_cell_168/BiasAdd:output:0@sequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
2sequential_4/simple_rnn_8/simple_rnn_cell_168/TanhTanh5sequential_4/simple_rnn_8/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P?
7sequential_4/simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)sequential_4/simple_rnn_8/TensorArrayV2_1TensorListReserve@sequential_4/simple_rnn_8/TensorArrayV2_1/element_shape:output:02sequential_4/simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???`
sequential_4/simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2sequential_4/simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????n
,sequential_4/simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_4/simple_rnn_8/whileWhile5sequential_4/simple_rnn_8/while/loop_counter:output:0;sequential_4/simple_rnn_8/while/maximum_iterations:output:0'sequential_4/simple_rnn_8/time:output:02sequential_4/simple_rnn_8/TensorArrayV2_1:handle:0(sequential_4/simple_rnn_8/zeros:output:02sequential_4/simple_rnn_8/strided_slice_1:output:0Qsequential_4/simple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0Lsequential_4_simple_rnn_8_simple_rnn_cell_168_matmul_readvariableop_resourceMsequential_4_simple_rnn_8_simple_rnn_cell_168_biasadd_readvariableop_resourceNsequential_4_simple_rnn_8_simple_rnn_cell_168_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_4_simple_rnn_8_while_body_11100783*9
cond1R/
-sequential_4_simple_rnn_8_while_cond_11100782*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
Jsequential_4/simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
<sequential_4/simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_4/simple_rnn_8/while:output:3Ssequential_4/simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0?
/sequential_4/simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1sequential_4/simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_4/simple_rnn_8/strided_slice_3StridedSliceEsequential_4/simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:08sequential_4/simple_rnn_8/strided_slice_3/stack:output:0:sequential_4/simple_rnn_8/strided_slice_3/stack_1:output:0:sequential_4/simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask
*sequential_4/simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
%sequential_4/simple_rnn_8/transpose_1	TransposeEsequential_4/simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P?
sequential_4/dropout_8/IdentityIdentity)sequential_4/simple_rnn_8/transpose_1:y:0*
T0*+
_output_shapes
:?????????Pw
sequential_4/simple_rnn_9/ShapeShape(sequential_4/dropout_8/Identity:output:0*
T0*
_output_shapes
:w
-sequential_4/simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_4/simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_4/simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'sequential_4/simple_rnn_9/strided_sliceStridedSlice(sequential_4/simple_rnn_9/Shape:output:06sequential_4/simple_rnn_9/strided_slice/stack:output:08sequential_4/simple_rnn_9/strided_slice/stack_1:output:08sequential_4/simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_4/simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
&sequential_4/simple_rnn_9/zeros/packedPack0sequential_4/simple_rnn_9/strided_slice:output:01sequential_4/simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_4/simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
sequential_4/simple_rnn_9/zerosFill/sequential_4/simple_rnn_9/zeros/packed:output:0.sequential_4/simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d}
(sequential_4/simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
#sequential_4/simple_rnn_9/transpose	Transpose(sequential_4/dropout_8/Identity:output:01sequential_4/simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:?????????Px
!sequential_4/simple_rnn_9/Shape_1Shape'sequential_4/simple_rnn_9/transpose:y:0*
T0*
_output_shapes
:y
/sequential_4/simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_4/simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_4/simple_rnn_9/strided_slice_1StridedSlice*sequential_4/simple_rnn_9/Shape_1:output:08sequential_4/simple_rnn_9/strided_slice_1/stack:output:0:sequential_4/simple_rnn_9/strided_slice_1/stack_1:output:0:sequential_4/simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
5sequential_4/simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
'sequential_4/simple_rnn_9/TensorArrayV2TensorListReserve>sequential_4/simple_rnn_9/TensorArrayV2/element_shape:output:02sequential_4/simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Osequential_4/simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Asequential_4/simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_4/simple_rnn_9/transpose:y:0Xsequential_4/simple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???y
/sequential_4/simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_4/simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_4/simple_rnn_9/strided_slice_2StridedSlice'sequential_4/simple_rnn_9/transpose:y:08sequential_4/simple_rnn_9/strided_slice_2/stack:output:0:sequential_4/simple_rnn_9/strided_slice_2/stack_1:output:0:sequential_4/simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
Csequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOpLsequential_4_simple_rnn_9_simple_rnn_cell_169_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
4sequential_4/simple_rnn_9/simple_rnn_cell_169/MatMulMatMul2sequential_4/simple_rnn_9/strided_slice_2:output:0Ksequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Dsequential_4/simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOpMsequential_4_simple_rnn_9_simple_rnn_cell_169_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
5sequential_4/simple_rnn_9/simple_rnn_cell_169/BiasAddBiasAdd>sequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul:product:0Lsequential_4/simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Esequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOpNsequential_4_simple_rnn_9_simple_rnn_cell_169_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
6sequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul_1MatMul(sequential_4/simple_rnn_9/zeros:output:0Msequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1sequential_4/simple_rnn_9/simple_rnn_cell_169/addAddV2>sequential_4/simple_rnn_9/simple_rnn_cell_169/BiasAdd:output:0@sequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
2sequential_4/simple_rnn_9/simple_rnn_cell_169/TanhTanh5sequential_4/simple_rnn_9/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d?
7sequential_4/simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
)sequential_4/simple_rnn_9/TensorArrayV2_1TensorListReserve@sequential_4/simple_rnn_9/TensorArrayV2_1/element_shape:output:02sequential_4/simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???`
sequential_4/simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : }
2sequential_4/simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????n
,sequential_4/simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential_4/simple_rnn_9/whileWhile5sequential_4/simple_rnn_9/while/loop_counter:output:0;sequential_4/simple_rnn_9/while/maximum_iterations:output:0'sequential_4/simple_rnn_9/time:output:02sequential_4/simple_rnn_9/TensorArrayV2_1:handle:0(sequential_4/simple_rnn_9/zeros:output:02sequential_4/simple_rnn_9/strided_slice_1:output:0Qsequential_4/simple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0Lsequential_4_simple_rnn_9_simple_rnn_cell_169_matmul_readvariableop_resourceMsequential_4_simple_rnn_9_simple_rnn_cell_169_biasadd_readvariableop_resourceNsequential_4_simple_rnn_9_simple_rnn_cell_169_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *9
body1R/
-sequential_4_simple_rnn_9_while_body_11100888*9
cond1R/
-sequential_4_simple_rnn_9_while_cond_11100887*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
Jsequential_4/simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
<sequential_4/simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_4/simple_rnn_9/while:output:3Ssequential_4/simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0?
/sequential_4/simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1sequential_4/simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_4/simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential_4/simple_rnn_9/strided_slice_3StridedSliceEsequential_4/simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:08sequential_4/simple_rnn_9/strided_slice_3/stack:output:0:sequential_4/simple_rnn_9/strided_slice_3/stack_1:output:0:sequential_4/simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask
*sequential_4/simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
%sequential_4/simple_rnn_9/transpose_1	TransposeEsequential_4/simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:03sequential_4/simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d?
sequential_4/dropout_9/IdentityIdentity2sequential_4/simple_rnn_9/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
sequential_4/dense_4/MatMulMatMul(sequential_4/dropout_9/Identity:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_4/dense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOpE^sequential_4/simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOpD^sequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOpF^sequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp ^sequential_4/simple_rnn_8/whileE^sequential_4/simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOpD^sequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOpF^sequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp ^sequential_4/simple_rnn_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2?
Dsequential_4/simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOpDsequential_4/simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp2?
Csequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOpCsequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp2?
Esequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOpEsequential_4/simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp2B
sequential_4/simple_rnn_8/whilesequential_4/simple_rnn_8/while2?
Dsequential_4/simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOpDsequential_4/simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp2?
Csequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOpCsequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp2?
Esequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOpEsequential_4/simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp2B
sequential_4/simple_rnn_9/whilesequential_4/simple_rnn_9/while:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_8_input
?-
?
while_body_11101920
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_169_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_169_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_169_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_169/MatMul/ReadVariableOp?1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_169_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_169/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_169/BiasAddBiasAdd*while/simple_rnn_cell_169/MatMul:product:08while/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_169/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_169/addAddV2*while/simple_rnn_cell_169/BiasAdd:output:0,while/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_169/TanhTanh!while/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_169/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_169/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_169/MatMul/ReadVariableOp2^while/simple_rnn_cell_169/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_169_biasadd_readvariableop_resource;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_169_matmul_readvariableop_resource:while_simple_rnn_cell_169_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_169/MatMul/ReadVariableOp/while/simple_rnn_cell_169/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?-
?
while_body_11101593
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_168_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_168_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_168_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_168/MatMul/ReadVariableOp?1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_168_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_168/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_168/BiasAddBiasAdd*while/simple_rnn_cell_168/MatMul:product:08while/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_168/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_168/addAddV2*while/simple_rnn_cell_168/BiasAdd:output:0,while/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_168/TanhTanh!while/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_168/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_168/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_168/MatMul/ReadVariableOp2^while/simple_rnn_cell_168/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_168_biasadd_readvariableop_resource;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_168_matmul_readvariableop_resource:while_simple_rnn_cell_168_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_168/MatMul/ReadVariableOp/while/simple_rnn_cell_168/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?E
?
-sequential_4_simple_rnn_9_while_body_11100888P
Lsequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_while_loop_counterV
Rsequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_while_maximum_iterations/
+sequential_4_simple_rnn_9_while_placeholder1
-sequential_4_simple_rnn_9_while_placeholder_11
-sequential_4_simple_rnn_9_while_placeholder_2O
Ksequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_strided_slice_1_0?
?sequential_4_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0f
Tsequential_4_simple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource_0:Pdc
Usequential_4_simple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource_0:dh
Vsequential_4_simple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0:dd,
(sequential_4_simple_rnn_9_while_identity.
*sequential_4_simple_rnn_9_while_identity_1.
*sequential_4_simple_rnn_9_while_identity_2.
*sequential_4_simple_rnn_9_while_identity_3.
*sequential_4_simple_rnn_9_while_identity_4M
Isequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_strided_slice_1?
?sequential_4_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_9_tensorarrayunstack_tensorlistfromtensord
Rsequential_4_simple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource:Pda
Ssequential_4_simple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource:df
Tsequential_4_simple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource:dd??Jsequential_4/simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp?Isequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp?Ksequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp?
Qsequential_4/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Csequential_4/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_4_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0+sequential_4_simple_rnn_9_while_placeholderZsequential_4/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
Isequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOpTsequential_4_simple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
:sequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMulMatMulJsequential_4/simple_rnn_9/while/TensorArrayV2Read/TensorListGetItem:item:0Qsequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Jsequential_4/simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOpUsequential_4_simple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
;sequential_4/simple_rnn_9/while/simple_rnn_cell_169/BiasAddBiasAddDsequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul:product:0Rsequential_4/simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Ksequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOpVsequential_4_simple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
<sequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul_1MatMul-sequential_4_simple_rnn_9_while_placeholder_2Ssequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
7sequential_4/simple_rnn_9/while/simple_rnn_cell_169/addAddV2Dsequential_4/simple_rnn_9/while/simple_rnn_cell_169/BiasAdd:output:0Fsequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
8sequential_4/simple_rnn_9/while/simple_rnn_cell_169/TanhTanh;sequential_4/simple_rnn_9/while/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d?
Dsequential_4/simple_rnn_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_4_simple_rnn_9_while_placeholder_1+sequential_4_simple_rnn_9_while_placeholder<sequential_4/simple_rnn_9/while/simple_rnn_cell_169/Tanh:y:0*
_output_shapes
: *
element_dtype0:???g
%sequential_4/simple_rnn_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential_4/simple_rnn_9/while/addAddV2+sequential_4_simple_rnn_9_while_placeholder.sequential_4/simple_rnn_9/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_4/simple_rnn_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
%sequential_4/simple_rnn_9/while/add_1AddV2Lsequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_while_loop_counter0sequential_4/simple_rnn_9/while/add_1/y:output:0*
T0*
_output_shapes
: ?
(sequential_4/simple_rnn_9/while/IdentityIdentity)sequential_4/simple_rnn_9/while/add_1:z:0%^sequential_4/simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ?
*sequential_4/simple_rnn_9/while/Identity_1IdentityRsequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_while_maximum_iterations%^sequential_4/simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ?
*sequential_4/simple_rnn_9/while/Identity_2Identity'sequential_4/simple_rnn_9/while/add:z:0%^sequential_4/simple_rnn_9/while/NoOp*
T0*
_output_shapes
: ?
*sequential_4/simple_rnn_9/while/Identity_3IdentityTsequential_4/simple_rnn_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_4/simple_rnn_9/while/NoOp*
T0*
_output_shapes
: :????
*sequential_4/simple_rnn_9/while/Identity_4Identity<sequential_4/simple_rnn_9/while/simple_rnn_cell_169/Tanh:y:0%^sequential_4/simple_rnn_9/while/NoOp*
T0*'
_output_shapes
:?????????d?
$sequential_4/simple_rnn_9/while/NoOpNoOpK^sequential_4/simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOpJ^sequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOpL^sequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_4_simple_rnn_9_while_identity1sequential_4/simple_rnn_9/while/Identity:output:0"a
*sequential_4_simple_rnn_9_while_identity_13sequential_4/simple_rnn_9/while/Identity_1:output:0"a
*sequential_4_simple_rnn_9_while_identity_23sequential_4/simple_rnn_9/while/Identity_2:output:0"a
*sequential_4_simple_rnn_9_while_identity_33sequential_4/simple_rnn_9/while/Identity_3:output:0"a
*sequential_4_simple_rnn_9_while_identity_43sequential_4/simple_rnn_9/while/Identity_4:output:0"?
Isequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_strided_slice_1Ksequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_strided_slice_1_0"?
Ssequential_4_simple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resourceUsequential_4_simple_rnn_9_while_simple_rnn_cell_169_biasadd_readvariableop_resource_0"?
Tsequential_4_simple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resourceVsequential_4_simple_rnn_9_while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0"?
Rsequential_4_simple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resourceTsequential_4_simple_rnn_9_while_simple_rnn_cell_169_matmul_readvariableop_resource_0"?
?sequential_4_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor?sequential_4_simple_rnn_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
Jsequential_4/simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOpJsequential_4/simple_rnn_9/while/simple_rnn_cell_169/BiasAdd/ReadVariableOp2?
Isequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOpIsequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul/ReadVariableOp2?
Ksequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOpKsequential_4/simple_rnn_9/while/simple_rnn_cell_169/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?	
?
/__inference_sequential_4_layer_call_fn_11101832
simple_rnn_8_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_11101813o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_8_input
?	
?
E__inference_dense_4_layer_call_and_return_conditional_losses_11103836

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
/__inference_simple_rnn_8_layer_call_fn_11102833
inputs_0
unknown:P
	unknown_0:P
	unknown_1:PP
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11101244|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
-sequential_4_simple_rnn_9_while_cond_11100887P
Lsequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_while_loop_counterV
Rsequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_while_maximum_iterations/
+sequential_4_simple_rnn_9_while_placeholder1
-sequential_4_simple_rnn_9_while_placeholder_11
-sequential_4_simple_rnn_9_while_placeholder_2R
Nsequential_4_simple_rnn_9_while_less_sequential_4_simple_rnn_9_strided_slice_1j
fsequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_while_cond_11100887___redundant_placeholder0j
fsequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_while_cond_11100887___redundant_placeholder1j
fsequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_while_cond_11100887___redundant_placeholder2j
fsequential_4_simple_rnn_9_while_sequential_4_simple_rnn_9_while_cond_11100887___redundant_placeholder3,
(sequential_4_simple_rnn_9_while_identity
?
$sequential_4/simple_rnn_9/while/LessLess+sequential_4_simple_rnn_9_while_placeholderNsequential_4_simple_rnn_9_while_less_sequential_4_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: 
(sequential_4/simple_rnn_9/while/IdentityIdentity(sequential_4/simple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_4_simple_rnn_9_while_identity1sequential_4/simple_rnn_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?H
?
!__inference__traced_save_11104076
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_simple_rnn_8_simple_rnn_cell_8_kernel_read_readvariableopN
Jsavev2_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_8_simple_rnn_cell_8_bias_read_readvariableopD
@savev2_simple_rnn_9_simple_rnn_cell_9_kernel_read_readvariableopN
Jsavev2_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_9_simple_rnn_cell_9_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_8_simple_rnn_cell_8_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_8_simple_rnn_cell_8_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_9_simple_rnn_cell_9_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_9_simple_rnn_cell_9_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_8_simple_rnn_cell_8_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_8_simple_rnn_cell_8_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_9_simple_rnn_cell_9_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_9_simple_rnn_cell_9_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_simple_rnn_8_simple_rnn_cell_8_kernel_read_readvariableopJsavev2_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_read_readvariableop>savev2_simple_rnn_8_simple_rnn_cell_8_bias_read_readvariableop@savev2_simple_rnn_9_simple_rnn_cell_9_kernel_read_readvariableopJsavev2_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_read_readvariableop>savev2_simple_rnn_9_simple_rnn_cell_9_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableopGsavev2_adam_simple_rnn_8_simple_rnn_cell_8_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_8_simple_rnn_cell_8_bias_m_read_readvariableopGsavev2_adam_simple_rnn_9_simple_rnn_cell_9_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_9_simple_rnn_cell_9_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableopGsavev2_adam_simple_rnn_8_simple_rnn_cell_8_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_8_simple_rnn_cell_8_bias_v_read_readvariableopGsavev2_adam_simple_rnn_9_simple_rnn_cell_9_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_9_simple_rnn_cell_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :d:: : : : : :P:PP:P:Pd:dd:d: : :d::P:PP:P:Pd:dd:d:d::P:PP:P:Pd:dd:d: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:P:$	 

_output_shapes

:PP: 


_output_shapes
:P:$ 

_output_shapes

:Pd:$ 

_output_shapes

:dd: 

_output_shapes
:d:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:Pd:$ 

_output_shapes

:dd: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:Pd:$ 

_output_shapes

:dd: 

_output_shapes
:d: 

_output_shapes
: 
?=
?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101986

inputsD
2simple_rnn_cell_169_matmul_readvariableop_resource:PdA
3simple_rnn_cell_169_biasadd_readvariableop_resource:dF
4simple_rnn_cell_169_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_169/BiasAdd/ReadVariableOp?)simple_rnn_cell_169/MatMul/ReadVariableOp?+simple_rnn_cell_169/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_169_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_169/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_169_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_169/BiasAddBiasAdd$simple_rnn_cell_169/MatMul:product:02simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_169_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_169/MatMul_1MatMulzeros:output:03simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_169/addAddV2$simple_rnn_cell_169/BiasAdd:output:0&simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_169/TanhTanhsimple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_169_matmul_readvariableop_resource3simple_rnn_cell_169_biasadd_readvariableop_resource4simple_rnn_cell_169_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11101920*
condR
while_cond_11101919*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_169/BiasAdd/ReadVariableOp*^simple_rnn_cell_169/MatMul/ReadVariableOp,^simple_rnn_cell_169/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_169/BiasAdd/ReadVariableOp*simple_rnn_cell_169/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_169/MatMul/ReadVariableOp)simple_rnn_cell_169/MatMul/ReadVariableOp2Z
+simple_rnn_cell_169/MatMul_1/ReadVariableOp+simple_rnn_cell_169/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?>
?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11102963
inputs_0D
2simple_rnn_cell_168_matmul_readvariableop_resource:PA
3simple_rnn_cell_168_biasadd_readvariableop_resource:PF
4simple_rnn_cell_168_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_168/BiasAdd/ReadVariableOp?)simple_rnn_cell_168/MatMul/ReadVariableOp?+simple_rnn_cell_168/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_168_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_168/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_168_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_168/BiasAddBiasAdd$simple_rnn_cell_168/MatMul:product:02simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_168_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_168/MatMul_1MatMulzeros:output:03simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_168/addAddV2$simple_rnn_cell_168/BiasAdd:output:0&simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_168/TanhTanhsimple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_168_matmul_readvariableop_resource3simple_rnn_cell_168_biasadd_readvariableop_resource4simple_rnn_cell_168_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11102897*
condR
while_cond_11102896*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????Pk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P?
NoOpNoOp+^simple_rnn_cell_168/BiasAdd/ReadVariableOp*^simple_rnn_cell_168/MatMul/ReadVariableOp,^simple_rnn_cell_168/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_168/BiasAdd/ReadVariableOp*simple_rnn_cell_168/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_168/MatMul/ReadVariableOp)simple_rnn_cell_168/MatMul/ReadVariableOp2Z
+simple_rnn_cell_168/MatMul_1/ReadVariableOp+simple_rnn_cell_168/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
/__inference_simple_rnn_9_layer_call_fn_11103325
inputs_0
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101377o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?
?
while_cond_11103723
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11103723___redundant_placeholder06
2while_while_cond_11103723___redundant_placeholder16
2while_while_cond_11103723___redundant_placeholder26
2while_while_cond_11103723___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?!
?
while_body_11101473
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_169_11101495_0:Pd2
$while_simple_rnn_cell_169_11101497_0:d6
$while_simple_rnn_cell_169_11101499_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_169_11101495:Pd0
"while_simple_rnn_cell_169_11101497:d4
"while_simple_rnn_cell_169_11101499:dd??1while/simple_rnn_cell_169/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_169/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_169_11101495_0$while_simple_rnn_cell_169_11101497_0$while_simple_rnn_cell_169_11101499_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11101421?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_169/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :????
while/Identity_4Identity:while/simple_rnn_cell_169/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_169/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_169_11101495$while_simple_rnn_cell_169_11101495_0"J
"while_simple_rnn_cell_169_11101497$while_simple_rnn_cell_169_11101497_0"J
"while_simple_rnn_cell_169_11101499$while_simple_rnn_cell_169_11101499_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_169/StatefulPartitionedCall1while/simple_rnn_cell_169/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_11102072
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11102072___redundant_placeholder06
2while_while_cond_11102072___redundant_placeholder16
2while_while_cond_11102072___redundant_placeholder26
2while_while_cond_11102072___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?4
?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11101244

inputs.
simple_rnn_cell_168_11101169:P*
simple_rnn_cell_168_11101171:P.
simple_rnn_cell_168_11101173:PP
identity??+simple_rnn_cell_168/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
+simple_rnn_cell_168/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_168_11101169simple_rnn_cell_168_11101171simple_rnn_cell_168_11101173*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11101129n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_168_11101169simple_rnn_cell_168_11101171simple_rnn_cell_168_11101173*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11101181*
condR
while_cond_11101180*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????Pk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P|
NoOpNoOp,^simple_rnn_cell_168/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_168/StatefulPartitionedCall+simple_rnn_cell_168/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?

?
 simple_rnn_9_while_cond_111027076
2simple_rnn_9_while_simple_rnn_9_while_loop_counter<
8simple_rnn_9_while_simple_rnn_9_while_maximum_iterations"
simple_rnn_9_while_placeholder$
 simple_rnn_9_while_placeholder_1$
 simple_rnn_9_while_placeholder_28
4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1P
Lsimple_rnn_9_while_simple_rnn_9_while_cond_11102707___redundant_placeholder0P
Lsimple_rnn_9_while_simple_rnn_9_while_cond_11102707___redundant_placeholder1P
Lsimple_rnn_9_while_simple_rnn_9_while_cond_11102707___redundant_placeholder2P
Lsimple_rnn_9_while_simple_rnn_9_while_cond_11102707___redundant_placeholder3
simple_rnn_9_while_identity
?
simple_rnn_9/while/LessLesssimple_rnn_9_while_placeholder4simple_rnn_9_while_less_simple_rnn_9_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_9/while/IdentityIdentitysimple_rnn_9/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_9_while_identity$simple_rnn_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?	
f
G__inference_dropout_9_layer_call_and_return_conditional_losses_11101862

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????dC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????do
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????di
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
while_cond_11101592
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11101592___redundant_placeholder06
2while_while_cond_11101592___redundant_placeholder16
2while_while_cond_11101592___redundant_placeholder26
2while_while_cond_11101592___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?>
?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11103071
inputs_0D
2simple_rnn_cell_168_matmul_readvariableop_resource:PA
3simple_rnn_cell_168_biasadd_readvariableop_resource:PF
4simple_rnn_cell_168_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_168/BiasAdd/ReadVariableOp?)simple_rnn_cell_168/MatMul/ReadVariableOp?+simple_rnn_cell_168/MatMul_1/ReadVariableOp?while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_168_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_168/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_168_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_168/BiasAddBiasAdd$simple_rnn_cell_168/MatMul:product:02simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_168_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_168/MatMul_1MatMulzeros:output:03simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_168/addAddV2$simple_rnn_cell_168/BiasAdd:output:0&simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_168/TanhTanhsimple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_168_matmul_readvariableop_resource3simple_rnn_cell_168_biasadd_readvariableop_resource4simple_rnn_cell_168_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11103005*
condR
while_cond_11103004*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????Pk
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :??????????????????P?
NoOpNoOp+^simple_rnn_cell_168/BiasAdd/ReadVariableOp*^simple_rnn_cell_168/MatMul/ReadVariableOp,^simple_rnn_cell_168/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_168/BiasAdd/ReadVariableOp*simple_rnn_cell_168/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_168/MatMul/ReadVariableOp)simple_rnn_cell_168/MatMul/ReadVariableOp2Z
+simple_rnn_cell_168/MatMul_1/ReadVariableOp+simple_rnn_cell_168/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?=
?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103790

inputsD
2simple_rnn_cell_169_matmul_readvariableop_resource:PdA
3simple_rnn_cell_169_biasadd_readvariableop_resource:dF
4simple_rnn_cell_169_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_169/BiasAdd/ReadVariableOp?)simple_rnn_cell_169/MatMul/ReadVariableOp?+simple_rnn_cell_169/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
)simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_169_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_169/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_169_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_169/BiasAddBiasAdd$simple_rnn_cell_169/MatMul:product:02simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_169_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_169/MatMul_1MatMulzeros:output:03simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_169/addAddV2$simple_rnn_cell_169/BiasAdd:output:0&simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_169/TanhTanhsimple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????dn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_169_matmul_readvariableop_resource3simple_rnn_cell_169_biasadd_readvariableop_resource4simple_rnn_cell_169_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11103724*
condR
while_cond_11103723*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_169/BiasAdd/ReadVariableOp*^simple_rnn_cell_169/MatMul/ReadVariableOp,^simple_rnn_cell_169/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_169/BiasAdd/ReadVariableOp*simple_rnn_cell_169/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_169/MatMul/ReadVariableOp)simple_rnn_cell_169/MatMul/ReadVariableOp2Z
+simple_rnn_cell_169/MatMul_1/ReadVariableOp+simple_rnn_cell_169/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?4
?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101377

inputs.
simple_rnn_cell_169_11101302:Pd*
simple_rnn_cell_169_11101304:d.
simple_rnn_cell_169_11101306:dd
identity??+simple_rnn_cell_169/StatefulPartitionedCall?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????dc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????PD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
+simple_rnn_cell_169/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_169_11101302simple_rnn_cell_169_11101304simple_rnn_cell_169_11101306*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????d:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11101301n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_169_11101302simple_rnn_cell_169_11101304simple_rnn_cell_169_11101306*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11101314*
condR
while_cond_11101313*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????d*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d|
NoOpNoOp,^simple_rnn_cell_169/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_169/StatefulPartitionedCall+simple_rnn_cell_169/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?
e
G__inference_dropout_9_layer_call_and_return_conditional_losses_11103805

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????d[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????d"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?	
f
G__inference_dropout_9_layer_call_and_return_conditional_losses_11103817

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:?????????dC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????do
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????di
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????dY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?-
?
while_body_11103508
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_169_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_169_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_169_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_169/MatMul/ReadVariableOp?1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_169_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_169/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_169/BiasAddBiasAdd*while/simple_rnn_cell_169/MatMul:product:08while/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_169/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_169/addAddV2*while/simple_rnn_cell_169/BiasAdd:output:0,while/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_169/TanhTanh!while/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_169/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_169/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_169/MatMul/ReadVariableOp2^while/simple_rnn_cell_169/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_169_biasadd_readvariableop_resource;while_simple_rnn_cell_169_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_169_matmul_1_readvariableop_resource<while_simple_rnn_cell_169_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_169_matmul_readvariableop_resource:while_simple_rnn_cell_169_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp0while/simple_rnn_cell_169/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_169/MatMul/ReadVariableOp/while/simple_rnn_cell_169/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp1while/simple_rnn_cell_169/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
: 
?	
?
&__inference_signature_wrapper_11102811
simple_rnn_8_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_11100961o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_8_input
?	
?
/__inference_sequential_4_layer_call_fn_11102313

inputs
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_4_layer_call_and_return_conditional_losses_11101813o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_9_layer_call_fn_11103795

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_11101794`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
??
?
$__inference__traced_restore_11104179
file_prefix1
assignvariableop_dense_4_kernel:d-
assignvariableop_1_dense_4_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: J
8assignvariableop_7_simple_rnn_8_simple_rnn_cell_8_kernel:PT
Bassignvariableop_8_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel:PPD
6assignvariableop_9_simple_rnn_8_simple_rnn_cell_8_bias:PK
9assignvariableop_10_simple_rnn_9_simple_rnn_cell_9_kernel:PdU
Cassignvariableop_11_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel:ddE
7assignvariableop_12_simple_rnn_9_simple_rnn_cell_9_bias:d#
assignvariableop_13_total: #
assignvariableop_14_count: ;
)assignvariableop_15_adam_dense_4_kernel_m:d5
'assignvariableop_16_adam_dense_4_bias_m:R
@assignvariableop_17_adam_simple_rnn_8_simple_rnn_cell_8_kernel_m:P\
Jassignvariableop_18_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_m:PPL
>assignvariableop_19_adam_simple_rnn_8_simple_rnn_cell_8_bias_m:PR
@assignvariableop_20_adam_simple_rnn_9_simple_rnn_cell_9_kernel_m:Pd\
Jassignvariableop_21_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_m:ddL
>assignvariableop_22_adam_simple_rnn_9_simple_rnn_cell_9_bias_m:d;
)assignvariableop_23_adam_dense_4_kernel_v:d5
'assignvariableop_24_adam_dense_4_bias_v:R
@assignvariableop_25_adam_simple_rnn_8_simple_rnn_cell_8_kernel_v:P\
Jassignvariableop_26_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_v:PPL
>assignvariableop_27_adam_simple_rnn_8_simple_rnn_cell_8_bias_v:PR
@assignvariableop_28_adam_simple_rnn_9_simple_rnn_cell_9_kernel_v:Pd\
Jassignvariableop_29_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_v:ddL
>assignvariableop_30_adam_simple_rnn_9_simple_rnn_cell_9_bias_v:d
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp8assignvariableop_7_simple_rnn_8_simple_rnn_cell_8_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpBassignvariableop_8_simple_rnn_8_simple_rnn_cell_8_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp6assignvariableop_9_simple_rnn_8_simple_rnn_cell_8_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp9assignvariableop_10_simple_rnn_9_simple_rnn_cell_9_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpCassignvariableop_11_simple_rnn_9_simple_rnn_cell_9_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp7assignvariableop_12_simple_rnn_9_simple_rnn_cell_9_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_4_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_4_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp@assignvariableop_17_adam_simple_rnn_8_simple_rnn_cell_8_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpJassignvariableop_18_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp>assignvariableop_19_adam_simple_rnn_8_simple_rnn_cell_8_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_simple_rnn_9_simple_rnn_cell_9_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpJassignvariableop_21_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp>assignvariableop_22_adam_simple_rnn_9_simple_rnn_cell_9_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_4_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_4_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_simple_rnn_8_simple_rnn_cell_8_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpJassignvariableop_26_adam_simple_rnn_8_simple_rnn_cell_8_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp>assignvariableop_27_adam_simple_rnn_8_simple_rnn_cell_8_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_simple_rnn_9_simple_rnn_cell_9_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpJassignvariableop_29_adam_simple_rnn_9_simple_rnn_cell_9_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_simple_rnn_9_simple_rnn_cell_9_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_32IdentityIdentity_31:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_32Identity_32:output:0*S
_input_shapesB
@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?9
?
 simple_rnn_8_while_body_111023766
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_25
1simple_rnn_8_while_simple_rnn_8_strided_slice_1_0q
msimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource_0:PV
Hsimple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource_0:P[
Isimple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0:PP
simple_rnn_8_while_identity!
simple_rnn_8_while_identity_1!
simple_rnn_8_while_identity_2!
simple_rnn_8_while_identity_3!
simple_rnn_8_while_identity_43
/simple_rnn_8_while_simple_rnn_8_strided_slice_1o
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource:PT
Fsimple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource:PY
Gsimple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource:PP??=simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp?<simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp?>simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp?
Dsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
6simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_8_while_placeholderMsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
<simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
-simple_rnn_8/while/simple_rnn_cell_168/MatMulMatMul=simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
=simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
.simple_rnn_8/while/simple_rnn_cell_168/BiasAddBiasAdd7simple_rnn_8/while/simple_rnn_cell_168/MatMul:product:0Esimple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
/simple_rnn_8/while/simple_rnn_cell_168/MatMul_1MatMul simple_rnn_8_while_placeholder_2Fsimple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_8/while/simple_rnn_cell_168/addAddV27simple_rnn_8/while/simple_rnn_cell_168/BiasAdd:output:09simple_rnn_8/while/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_8/while/simple_rnn_cell_168/TanhTanh.simple_rnn_8/while/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_8_while_placeholder_1simple_rnn_8_while_placeholder/simple_rnn_8/while/simple_rnn_cell_168/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_8/while/addAddV2simple_rnn_8_while_placeholder!simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_8/while/add_1AddV22simple_rnn_8_while_simple_rnn_8_while_loop_counter#simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/add_1:z:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_8/while/Identity_1Identity8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_8/while/Identity_2Identitysimple_rnn_8/while/add:z:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_8/while/Identity_3IdentityGsimple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_8/while/Identity_4Identity/simple_rnn_8/while/simple_rnn_cell_168/Tanh:y:0^simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_8/while/NoOpNoOp>^simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp=^simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp?^simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0"G
simple_rnn_8_while_identity_1&simple_rnn_8/while/Identity_1:output:0"G
simple_rnn_8_while_identity_2&simple_rnn_8/while/Identity_2:output:0"G
simple_rnn_8_while_identity_3&simple_rnn_8/while/Identity_3:output:0"G
simple_rnn_8_while_identity_4&simple_rnn_8/while/Identity_4:output:0"d
/simple_rnn_8_while_simple_rnn_8_strided_slice_11simple_rnn_8_while_simple_rnn_8_strided_slice_1_0"?
Fsimple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resourceHsimple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource_0"?
Gsimple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resourceIsimple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0"?
Esimple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resourceGsimple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource_0"?
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensormsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2~
=simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp=simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp2|
<simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp<simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp2?
>simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp>simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?9
?
 simple_rnn_8_while_body_111025966
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_25
1simple_rnn_8_while_simple_rnn_8_strided_slice_1_0q
msimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0Y
Gsimple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource_0:PV
Hsimple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource_0:P[
Isimple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0:PP
simple_rnn_8_while_identity!
simple_rnn_8_while_identity_1!
simple_rnn_8_while_identity_2!
simple_rnn_8_while_identity_3!
simple_rnn_8_while_identity_43
/simple_rnn_8_while_simple_rnn_8_strided_slice_1o
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensorW
Esimple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource:PT
Fsimple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource:PY
Gsimple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource:PP??=simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp?<simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp?>simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp?
Dsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
6simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_8_while_placeholderMsimple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
<simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOpGsimple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
-simple_rnn_8/while/simple_rnn_cell_168/MatMulMatMul=simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Dsimple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
=simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOpHsimple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
.simple_rnn_8/while/simple_rnn_cell_168/BiasAddBiasAdd7simple_rnn_8/while/simple_rnn_cell_168/MatMul:product:0Esimple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOpIsimple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
/simple_rnn_8/while/simple_rnn_cell_168/MatMul_1MatMul simple_rnn_8_while_placeholder_2Fsimple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_8/while/simple_rnn_cell_168/addAddV27simple_rnn_8/while/simple_rnn_cell_168/BiasAdd:output:09simple_rnn_8/while/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_8/while/simple_rnn_cell_168/TanhTanh.simple_rnn_8/while/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_8_while_placeholder_1simple_rnn_8_while_placeholder/simple_rnn_8/while/simple_rnn_cell_168/Tanh:y:0*
_output_shapes
: *
element_dtype0:???Z
simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_8/while/addAddV2simple_rnn_8_while_placeholder!simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_8/while/add_1AddV22simple_rnn_8_while_simple_rnn_8_while_loop_counter#simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/add_1:z:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_8/while/Identity_1Identity8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_8/while/Identity_2Identitysimple_rnn_8/while/add:z:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_8/while/Identity_3IdentityGsimple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_8/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_8/while/Identity_4Identity/simple_rnn_8/while/simple_rnn_cell_168/Tanh:y:0^simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_8/while/NoOpNoOp>^simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp=^simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp?^simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0"G
simple_rnn_8_while_identity_1&simple_rnn_8/while/Identity_1:output:0"G
simple_rnn_8_while_identity_2&simple_rnn_8/while/Identity_2:output:0"G
simple_rnn_8_while_identity_3&simple_rnn_8/while/Identity_3:output:0"G
simple_rnn_8_while_identity_4&simple_rnn_8/while/Identity_4:output:0"d
/simple_rnn_8_while_simple_rnn_8_strided_slice_11simple_rnn_8_while_simple_rnn_8_strided_slice_1_0"?
Fsimple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resourceHsimple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource_0"?
Gsimple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resourceIsimple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0"?
Esimple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resourceGsimple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource_0"?
ksimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensormsimple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2~
=simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp=simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp2|
<simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp<simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp2?
>simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp>simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
??
?	
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102788

inputsQ
?simple_rnn_8_simple_rnn_cell_168_matmul_readvariableop_resource:PN
@simple_rnn_8_simple_rnn_cell_168_biasadd_readvariableop_resource:PS
Asimple_rnn_8_simple_rnn_cell_168_matmul_1_readvariableop_resource:PPQ
?simple_rnn_9_simple_rnn_cell_169_matmul_readvariableop_resource:PdN
@simple_rnn_9_simple_rnn_cell_169_biasadd_readvariableop_resource:dS
Asimple_rnn_9_simple_rnn_cell_169_matmul_1_readvariableop_resource:dd8
&dense_4_matmul_readvariableop_resource:d5
'dense_4_biasadd_readvariableop_resource:
identity??dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?7simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp?6simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp?8simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp?simple_rnn_8/while?7simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp?6simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp?8simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp?simple_rnn_9/whileH
simple_rnn_8/ShapeShapeinputs*
T0*
_output_shapes
:j
 simple_rnn_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_8/strided_sliceStridedSlicesimple_rnn_8/Shape:output:0)simple_rnn_8/strided_slice/stack:output:0+simple_rnn_8/strided_slice/stack_1:output:0+simple_rnn_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_8/zeros/packedPack#simple_rnn_8/strided_slice:output:0$simple_rnn_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_8/zerosFill"simple_rnn_8/zeros/packed:output:0!simple_rnn_8/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pp
simple_rnn_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_8/transpose	Transposeinputs$simple_rnn_8/transpose/perm:output:0*
T0*+
_output_shapes
:?????????^
simple_rnn_8/Shape_1Shapesimple_rnn_8/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_8/strided_slice_1StridedSlicesimple_rnn_8/Shape_1:output:0+simple_rnn_8/strided_slice_1/stack:output:0-simple_rnn_8/strided_slice_1/stack_1:output:0-simple_rnn_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_8/TensorArrayV2TensorListReserve1simple_rnn_8/TensorArrayV2/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
4simple_rnn_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_8/transpose:y:0Ksimple_rnn_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_8/strided_slice_2StridedSlicesimple_rnn_8/transpose:y:0+simple_rnn_8/strided_slice_2/stack:output:0-simple_rnn_8/strided_slice_2/stack_1:output:0-simple_rnn_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
6simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp?simple_rnn_8_simple_rnn_cell_168_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
'simple_rnn_8/simple_rnn_cell_168/MatMulMatMul%simple_rnn_8/strided_slice_2:output:0>simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
7simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_8_simple_rnn_cell_168_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
(simple_rnn_8/simple_rnn_cell_168/BiasAddBiasAdd1simple_rnn_8/simple_rnn_cell_168/MatMul:product:0?simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_8_simple_rnn_cell_168_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
)simple_rnn_8/simple_rnn_cell_168/MatMul_1MatMulsimple_rnn_8/zeros:output:0@simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
$simple_rnn_8/simple_rnn_cell_168/addAddV21simple_rnn_8/simple_rnn_cell_168/BiasAdd:output:03simple_rnn_8/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_8/simple_rnn_cell_168/TanhTanh(simple_rnn_8/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P{
*simple_rnn_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_8/TensorArrayV2_1TensorListReserve3simple_rnn_8/TensorArrayV2_1/element_shape:output:0%simple_rnn_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_8/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_8/whileWhile(simple_rnn_8/while/loop_counter:output:0.simple_rnn_8/while/maximum_iterations:output:0simple_rnn_8/time:output:0%simple_rnn_8/TensorArrayV2_1:handle:0simple_rnn_8/zeros:output:0%simple_rnn_8/strided_slice_1:output:0Dsimple_rnn_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_8_simple_rnn_cell_168_matmul_readvariableop_resource@simple_rnn_8_simple_rnn_cell_168_biasadd_readvariableop_resourceAsimple_rnn_8_simple_rnn_cell_168_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_8_while_body_11102596*,
cond$R"
 simple_rnn_8_while_cond_11102595*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
=simple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
/simple_rnn_8/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_8/while:output:3Fsimple_rnn_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0u
"simple_rnn_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_8/strided_slice_3StridedSlice8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_8/strided_slice_3/stack:output:0-simple_rnn_8/strided_slice_3/stack_1:output:0-simple_rnn_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maskr
simple_rnn_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_8/transpose_1	Transpose8simple_rnn_8/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_8/dropout/MulMulsimple_rnn_8/transpose_1:y:0 dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:?????????Pc
dropout_8/dropout/ShapeShapesimple_rnn_8/transpose_1:y:0*
T0*
_output_shapes
:?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????P?
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????P?
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
simple_rnn_9/ShapeShapedropout_8/dropout/Mul_1:z:0*
T0*
_output_shapes
:j
 simple_rnn_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_9/strided_sliceStridedSlicesimple_rnn_9/Shape:output:0)simple_rnn_9/strided_slice/stack:output:0+simple_rnn_9/strided_slice/stack_1:output:0+simple_rnn_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_9/zeros/packedPack#simple_rnn_9/strided_slice:output:0$simple_rnn_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_9/zerosFill"simple_rnn_9/zeros/packed:output:0!simple_rnn_9/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dp
simple_rnn_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_9/transpose	Transposedropout_8/dropout/Mul_1:z:0$simple_rnn_9/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P^
simple_rnn_9/Shape_1Shapesimple_rnn_9/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_9/strided_slice_1StridedSlicesimple_rnn_9/Shape_1:output:0+simple_rnn_9/strided_slice_1/stack:output:0-simple_rnn_9/strided_slice_1/stack_1:output:0-simple_rnn_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(simple_rnn_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_9/TensorArrayV2TensorListReserve1simple_rnn_9/TensorArrayV2/element_shape:output:0%simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Bsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
4simple_rnn_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_9/transpose:y:0Ksimple_rnn_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???l
"simple_rnn_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_9/strided_slice_2StridedSlicesimple_rnn_9/transpose:y:0+simple_rnn_9/strided_slice_2/stack:output:0-simple_rnn_9/strided_slice_2/stack_1:output:0-simple_rnn_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
6simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOpReadVariableOp?simple_rnn_9_simple_rnn_cell_169_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
'simple_rnn_9/simple_rnn_cell_169/MatMulMatMul%simple_rnn_9/strided_slice_2:output:0>simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
7simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOpReadVariableOp@simple_rnn_9_simple_rnn_cell_169_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
(simple_rnn_9/simple_rnn_cell_169/BiasAddBiasAdd1simple_rnn_9/simple_rnn_cell_169/MatMul:product:0?simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOpReadVariableOpAsimple_rnn_9_simple_rnn_cell_169_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
)simple_rnn_9/simple_rnn_cell_169/MatMul_1MatMulsimple_rnn_9/zeros:output:0@simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
$simple_rnn_9/simple_rnn_cell_169/addAddV21simple_rnn_9/simple_rnn_cell_169/BiasAdd:output:03simple_rnn_9/simple_rnn_cell_169/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_9/simple_rnn_cell_169/TanhTanh(simple_rnn_9/simple_rnn_cell_169/add:z:0*
T0*'
_output_shapes
:?????????d{
*simple_rnn_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_9/TensorArrayV2_1TensorListReserve3simple_rnn_9/TensorArrayV2_1/element_shape:output:0%simple_rnn_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???S
simple_rnn_9/timeConst*
_output_shapes
: *
dtype0*
value	B : p
%simple_rnn_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????a
simple_rnn_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_9/whileWhile(simple_rnn_9/while/loop_counter:output:0.simple_rnn_9/while/maximum_iterations:output:0simple_rnn_9/time:output:0%simple_rnn_9/TensorArrayV2_1:handle:0simple_rnn_9/zeros:output:0%simple_rnn_9/strided_slice_1:output:0Dsimple_rnn_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0?simple_rnn_9_simple_rnn_cell_169_matmul_readvariableop_resource@simple_rnn_9_simple_rnn_cell_169_biasadd_readvariableop_resourceAsimple_rnn_9_simple_rnn_cell_169_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????d: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *,
body$R"
 simple_rnn_9_while_body_11102708*,
cond$R"
 simple_rnn_9_while_cond_11102707*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
=simple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
/simple_rnn_9/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_9/while:output:3Fsimple_rnn_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0u
"simple_rnn_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????n
$simple_rnn_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_9/strided_slice_3StridedSlice8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_9/strided_slice_3/stack:output:0-simple_rnn_9/strided_slice_3/stack_1:output:0-simple_rnn_9/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_maskr
simple_rnn_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_9/transpose_1	Transpose8simple_rnn_9/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_9/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d\
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_9/dropout/MulMul%simple_rnn_9/strided_slice_3:output:0 dropout_9/dropout/Const:output:0*
T0*'
_output_shapes
:?????????dl
dropout_9/dropout/ShapeShape%simple_rnn_9/strided_slice_3:output:0*
T0*
_output_shapes
:?
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0e
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_4/MatMulMatMuldropout_9/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp8^simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp7^simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp9^simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp^simple_rnn_8/while8^simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp7^simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp9^simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp^simple_rnn_9/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2r
7simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp7simple_rnn_8/simple_rnn_cell_168/BiasAdd/ReadVariableOp2p
6simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp6simple_rnn_8/simple_rnn_cell_168/MatMul/ReadVariableOp2t
8simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp8simple_rnn_8/simple_rnn_cell_168/MatMul_1/ReadVariableOp2(
simple_rnn_8/whilesimple_rnn_8/while2r
7simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp7simple_rnn_9/simple_rnn_cell_169/BiasAdd/ReadVariableOp2p
6simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp6simple_rnn_9/simple_rnn_cell_169/MatMul/ReadVariableOp2t
8simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp8simple_rnn_9/simple_rnn_cell_169/MatMul_1/ReadVariableOp2(
simple_rnn_9/whilesimple_rnn_9/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102261
simple_rnn_8_input'
simple_rnn_8_11102239:P#
simple_rnn_8_11102241:P'
simple_rnn_8_11102243:PP'
simple_rnn_9_11102247:Pd#
simple_rnn_9_11102249:d'
simple_rnn_9_11102251:dd"
dense_4_11102255:d
dense_4_11102257:
identity??dense_4/StatefulPartitionedCall?$simple_rnn_8/StatefulPartitionedCall?$simple_rnn_9/StatefulPartitionedCall?
$simple_rnn_8/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_8_inputsimple_rnn_8_11102239simple_rnn_8_11102241simple_rnn_8_11102243*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11101659?
dropout_8/PartitionedCallPartitionedCall-simple_rnn_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_11101672?
$simple_rnn_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0simple_rnn_9_11102247simple_rnn_9_11102249simple_rnn_9_11102251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101781?
dropout_9/PartitionedCallPartitionedCall-simple_rnn_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_11101794?
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_4_11102255dense_4_11102257*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_11101806w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_4/StatefulPartitionedCall%^simple_rnn_8/StatefulPartitionedCall%^simple_rnn_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2L
$simple_rnn_8/StatefulPartitionedCall$simple_rnn_8/StatefulPartitionedCall2L
$simple_rnn_9/StatefulPartitionedCall$simple_rnn_9/StatefulPartitionedCall:_ [
+
_output_shapes
:?????????
,
_user_specified_namesimple_rnn_8_input
?=
?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11103179

inputsD
2simple_rnn_cell_168_matmul_readvariableop_resource:PA
3simple_rnn_cell_168_biasadd_readvariableop_resource:PF
4simple_rnn_cell_168_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_168/BiasAdd/ReadVariableOp?)simple_rnn_cell_168/MatMul/ReadVariableOp?+simple_rnn_cell_168/MatMul_1/ReadVariableOp?while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ps
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
)simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_168_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_168/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_168_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_168/BiasAddBiasAdd$simple_rnn_cell_168/MatMul:product:02simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_168_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_168/MatMul_1MatMulzeros:output:03simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_168/addAddV2$simple_rnn_cell_168/BiasAdd:output:0&simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_168/TanhTanhsimple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????Pn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_168_matmul_readvariableop_resource3simple_rnn_cell_168_biasadd_readvariableop_resource4simple_rnn_cell_168_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????P: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_11103113*
condR
while_cond_11103112*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_168/BiasAdd/ReadVariableOp*^simple_rnn_cell_168/MatMul/ReadVariableOp,^simple_rnn_cell_168/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_168/BiasAdd/ReadVariableOp*simple_rnn_cell_168/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_168/MatMul/ReadVariableOp)simple_rnn_cell_168/MatMul/ReadVariableOp2Z
+simple_rnn_cell_168/MatMul_1/ReadVariableOp+simple_rnn_cell_168/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
-sequential_4_simple_rnn_8_while_body_11100783P
Lsequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_while_loop_counterV
Rsequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_while_maximum_iterations/
+sequential_4_simple_rnn_8_while_placeholder1
-sequential_4_simple_rnn_8_while_placeholder_11
-sequential_4_simple_rnn_8_while_placeholder_2O
Ksequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_strided_slice_1_0?
?sequential_4_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0f
Tsequential_4_simple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource_0:Pc
Usequential_4_simple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource_0:Ph
Vsequential_4_simple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0:PP,
(sequential_4_simple_rnn_8_while_identity.
*sequential_4_simple_rnn_8_while_identity_1.
*sequential_4_simple_rnn_8_while_identity_2.
*sequential_4_simple_rnn_8_while_identity_3.
*sequential_4_simple_rnn_8_while_identity_4M
Isequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_strided_slice_1?
?sequential_4_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_8_tensorarrayunstack_tensorlistfromtensord
Rsequential_4_simple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource:Pa
Ssequential_4_simple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource:Pf
Tsequential_4_simple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource:PP??Jsequential_4/simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp?Isequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp?Ksequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp?
Qsequential_4/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Csequential_4/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_4_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0+sequential_4_simple_rnn_8_while_placeholderZsequential_4/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
Isequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOpTsequential_4_simple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
:sequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMulMatMulJsequential_4/simple_rnn_8/while/TensorArrayV2Read/TensorListGetItem:item:0Qsequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Jsequential_4/simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOpUsequential_4_simple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
;sequential_4/simple_rnn_8/while/simple_rnn_cell_168/BiasAddBiasAddDsequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul:product:0Rsequential_4/simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Ksequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOpVsequential_4_simple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
<sequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul_1MatMul-sequential_4_simple_rnn_8_while_placeholder_2Ssequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
7sequential_4/simple_rnn_8/while/simple_rnn_cell_168/addAddV2Dsequential_4/simple_rnn_8/while/simple_rnn_cell_168/BiasAdd:output:0Fsequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
8sequential_4/simple_rnn_8/while/simple_rnn_cell_168/TanhTanh;sequential_4/simple_rnn_8/while/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P?
Dsequential_4/simple_rnn_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_4_simple_rnn_8_while_placeholder_1+sequential_4_simple_rnn_8_while_placeholder<sequential_4/simple_rnn_8/while/simple_rnn_cell_168/Tanh:y:0*
_output_shapes
: *
element_dtype0:???g
%sequential_4/simple_rnn_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
#sequential_4/simple_rnn_8/while/addAddV2+sequential_4_simple_rnn_8_while_placeholder.sequential_4/simple_rnn_8/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_4/simple_rnn_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
%sequential_4/simple_rnn_8/while/add_1AddV2Lsequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_while_loop_counter0sequential_4/simple_rnn_8/while/add_1/y:output:0*
T0*
_output_shapes
: ?
(sequential_4/simple_rnn_8/while/IdentityIdentity)sequential_4/simple_rnn_8/while/add_1:z:0%^sequential_4/simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ?
*sequential_4/simple_rnn_8/while/Identity_1IdentityRsequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_while_maximum_iterations%^sequential_4/simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ?
*sequential_4/simple_rnn_8/while/Identity_2Identity'sequential_4/simple_rnn_8/while/add:z:0%^sequential_4/simple_rnn_8/while/NoOp*
T0*
_output_shapes
: ?
*sequential_4/simple_rnn_8/while/Identity_3IdentityTsequential_4/simple_rnn_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_4/simple_rnn_8/while/NoOp*
T0*
_output_shapes
: :????
*sequential_4/simple_rnn_8/while/Identity_4Identity<sequential_4/simple_rnn_8/while/simple_rnn_cell_168/Tanh:y:0%^sequential_4/simple_rnn_8/while/NoOp*
T0*'
_output_shapes
:?????????P?
$sequential_4/simple_rnn_8/while/NoOpNoOpK^sequential_4/simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOpJ^sequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOpL^sequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_4_simple_rnn_8_while_identity1sequential_4/simple_rnn_8/while/Identity:output:0"a
*sequential_4_simple_rnn_8_while_identity_13sequential_4/simple_rnn_8/while/Identity_1:output:0"a
*sequential_4_simple_rnn_8_while_identity_23sequential_4/simple_rnn_8/while/Identity_2:output:0"a
*sequential_4_simple_rnn_8_while_identity_33sequential_4/simple_rnn_8/while/Identity_3:output:0"a
*sequential_4_simple_rnn_8_while_identity_43sequential_4/simple_rnn_8/while/Identity_4:output:0"?
Isequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_strided_slice_1Ksequential_4_simple_rnn_8_while_sequential_4_simple_rnn_8_strided_slice_1_0"?
Ssequential_4_simple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resourceUsequential_4_simple_rnn_8_while_simple_rnn_cell_168_biasadd_readvariableop_resource_0"?
Tsequential_4_simple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resourceVsequential_4_simple_rnn_8_while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0"?
Rsequential_4_simple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resourceTsequential_4_simple_rnn_8_while_simple_rnn_cell_168_matmul_readvariableop_resource_0"?
?sequential_4_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor?sequential_4_simple_rnn_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_simple_rnn_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
Jsequential_4/simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOpJsequential_4/simple_rnn_8/while/simple_rnn_cell_168/BiasAdd/ReadVariableOp2?
Isequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOpIsequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul/ReadVariableOp2?
Ksequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOpKsequential_4/simple_rnn_8/while/simple_rnn_cell_168/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
?
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11101009

inputs

states0
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P2
 matmul_1_readvariableop_resource:PP
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Px
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????PG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????PW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????PY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????P?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????P
 
_user_specified_namestates
?
?
while_cond_11103507
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11103507___redundant_placeholder06
2while_while_cond_11103507___redundant_placeholder16
2while_while_cond_11103507___redundant_placeholder26
2while_while_cond_11103507___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?

?
6__inference_simple_rnn_cell_168_layer_call_fn_11103864

inputs
states_0
unknown:P
	unknown_0:P
	unknown_1:PP
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????P:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11101129o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Pq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????P
"
_user_specified_name
states/0
?

f
G__inference_dropout_8_layer_call_and_return_conditional_losses_11103314

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
while_cond_11103615
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11103615___redundant_placeholder06
2while_while_cond_11103615___redundant_placeholder16
2while_while_cond_11103615___redundant_placeholder26
2while_while_cond_11103615___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?	
?
E__inference_dense_4_layer_call_and_return_conditional_losses_11101806

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11103898

inputs
states_00
matmul_readvariableop_resource:P-
biasadd_readvariableop_resource:P2
 matmul_1_readvariableop_resource:PP
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Px
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Pd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????PG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????PW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????PY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????P?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????:?????????P: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????P
"
_user_specified_name
states/0
?
?
/__inference_simple_rnn_9_layer_call_fn_11103336
inputs_0
unknown:Pd
	unknown_0:d
	unknown_1:dd
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11101536o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?
?
while_cond_11103004
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11103004___redundant_placeholder06
2while_while_cond_11103004___redundant_placeholder16
2while_while_cond_11103004___redundant_placeholder26
2while_while_cond_11103004___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?
?
*__inference_dense_4_layer_call_fn_11103826

inputs
unknown:d
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_4_layer_call_and_return_conditional_losses_11101806o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?

?
 simple_rnn_8_while_cond_111025956
2simple_rnn_8_while_simple_rnn_8_while_loop_counter<
8simple_rnn_8_while_simple_rnn_8_while_maximum_iterations"
simple_rnn_8_while_placeholder$
 simple_rnn_8_while_placeholder_1$
 simple_rnn_8_while_placeholder_28
4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1P
Lsimple_rnn_8_while_simple_rnn_8_while_cond_11102595___redundant_placeholder0P
Lsimple_rnn_8_while_simple_rnn_8_while_cond_11102595___redundant_placeholder1P
Lsimple_rnn_8_while_simple_rnn_8_while_cond_11102595___redundant_placeholder2P
Lsimple_rnn_8_while_simple_rnn_8_while_cond_11102595___redundant_placeholder3
simple_rnn_8_while_identity
?
simple_rnn_8/while/LessLesssimple_rnn_8_while_placeholder4simple_rnn_8_while_less_simple_rnn_8_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_8/while/IdentityIdentitysimple_rnn_8/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_8_while_identity$simple_rnn_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????P: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
:
?-
?
while_body_11103005
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_168_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_168_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_168_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_168/MatMul/ReadVariableOp?1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_168/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_168_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_168/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_168/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_168/BiasAddBiasAdd*while/simple_rnn_cell_168/MatMul:product:08while/simple_rnn_cell_168/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_168/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_168/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_168/addAddV2*while/simple_rnn_cell_168/BiasAdd:output:0,while/simple_rnn_cell_168/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_168/TanhTanh!while/simple_rnn_cell_168/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_168/Tanh:y:0*
_output_shapes
: *
element_dtype0:???M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: ?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :???
while/Identity_4Identity"while/simple_rnn_cell_168/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_168/MatMul/ReadVariableOp2^while/simple_rnn_cell_168/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_168_biasadd_readvariableop_resource;while_simple_rnn_cell_168_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_168_matmul_1_readvariableop_resource<while_simple_rnn_cell_168_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_168_matmul_readvariableop_resource:while_simple_rnn_cell_168_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp0while/simple_rnn_cell_168/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_168/MatMul/ReadVariableOp/while/simple_rnn_cell_168/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp1while/simple_rnn_cell_168/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????P:

_output_shapes
: :

_output_shapes
: 
?
e
,__inference_dropout_9_layer_call_fn_11103800

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_11101862o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????d22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
e
G__inference_dropout_8_layer_call_and_return_conditional_losses_11103302

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
while_cond_11101313
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11101313___redundant_placeholder06
2while_while_cond_11101313___redundant_placeholder16
2while_while_cond_11101313___redundant_placeholder26
2while_while_cond_11101313___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :?????????d: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????d:

_output_shapes
: :

_output_shapes
:
?
?
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11101301

inputs

states0
matmul_readvariableop_resource:Pd-
biasadd_readvariableop_resource:d2
 matmul_1_readvariableop_resource:dd
identity

identity_1??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dx
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????dd
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:?????????dG
TanhTanhadd:z:0*
T0*'
_output_shapes
:?????????dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????dY

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes.
,:?????????P:?????????d: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????P
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????d
 
_user_specified_namestates"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
simple_rnn_8_input?
$serving_default_simple_rnn_8_input:0?????????;
dense_40
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer

signatures
#_self_saveable_object_factories
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_sequential
?
cell

state_spec
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
?
!cell
"
state_spec
##_self_saveable_object_factories
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
?
#*_self_saveable_object_factories
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/_random_generator
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
?

2kernel
3bias
#4_self_saveable_object_factories
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
?
;iter

<beta_1

=beta_2
	>decay
?learning_rate2m?3m?Am?Bm?Cm?Dm?Em?Fm?2v?3v?Av?Bv?Cv?Dv?Ev?Fv?"
	optimizer
,
@serving_default"
signature_map
 "
trackable_dict_wrapper
X
A0
B1
C2
D3
E4
F5
26
37"
trackable_list_wrapper
X
A0
B1
C2
D3
E4
F5
26
37"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_sequential_4_layer_call_fn_11101832
/__inference_sequential_4_layer_call_fn_11102313
/__inference_sequential_4_layer_call_fn_11102334
/__inference_sequential_4_layer_call_fn_11102236?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102554
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102788
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102261
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102286?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_11100961simple_rnn_8_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?

Akernel
Brecurrent_kernel
Cbias
#L_self_saveable_object_factories
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q_random_generator
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
A0
B1
C2"
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Tstates
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_simple_rnn_8_layer_call_fn_11102822
/__inference_simple_rnn_8_layer_call_fn_11102833
/__inference_simple_rnn_8_layer_call_fn_11102844
/__inference_simple_rnn_8_layer_call_fn_11102855?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11102963
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11103071
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11103179
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11103287?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
,__inference_dropout_8_layer_call_fn_11103292
,__inference_dropout_8_layer_call_fn_11103297?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_8_layer_call_and_return_conditional_losses_11103302
G__inference_dropout_8_layer_call_and_return_conditional_losses_11103314?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?

Dkernel
Erecurrent_kernel
Fbias
#__self_saveable_object_factories
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d_random_generator
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
D0
E1
F2"
trackable_list_wrapper
5
D0
E1
F2"
trackable_list_wrapper
 "
trackable_list_wrapper
?

gstates
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_simple_rnn_9_layer_call_fn_11103325
/__inference_simple_rnn_9_layer_call_fn_11103336
/__inference_simple_rnn_9_layer_call_fn_11103347
/__inference_simple_rnn_9_layer_call_fn_11103358?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103466
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103574
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103682
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103790?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
+	variables
,trainable_variables
-regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
,__inference_dropout_9_layer_call_fn_11103795
,__inference_dropout_9_layer_call_fn_11103800?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_9_layer_call_and_return_conditional_losses_11103805
G__inference_dropout_9_layer_call_and_return_conditional_losses_11103817?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 :d2dense_4/kernel
:2dense_4/bias
 "
trackable_dict_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_dense_4_layer_call_fn_11103826?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_4_layer_call_and_return_conditional_losses_11103836?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
&__inference_signature_wrapper_11102811simple_rnn_8_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
7:5P2%simple_rnn_8/simple_rnn_cell_8/kernel
A:?PP2/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel
1:/P2#simple_rnn_8/simple_rnn_cell_8/bias
7:5Pd2%simple_rnn_9/simple_rnn_cell_9/kernel
A:?dd2/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel
1:/d2#simple_rnn_9/simple_rnn_cell_9/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
w0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
5
A0
B1
C2"
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
6__inference_simple_rnn_cell_168_layer_call_fn_11103850
6__inference_simple_rnn_cell_168_layer_call_fn_11103864?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11103881
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11103898?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
5
D0
E1
F2"
trackable_list_wrapper
5
D0
E1
F2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
`	variables
atrainable_variables
bregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
?2?
6__inference_simple_rnn_cell_169_layer_call_fn_11103912
6__inference_simple_rnn_cell_169_layer_call_fn_11103926?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11103943
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11103960?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
!0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
%:#d2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
<::P2,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/m
F:DPP26Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/m
6:4P2*Adam/simple_rnn_8/simple_rnn_cell_8/bias/m
<::Pd2,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/m
F:Ddd26Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/m
6:4d2*Adam/simple_rnn_9/simple_rnn_cell_9/bias/m
%:#d2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
<::P2,Adam/simple_rnn_8/simple_rnn_cell_8/kernel/v
F:DPP26Adam/simple_rnn_8/simple_rnn_cell_8/recurrent_kernel/v
6:4P2*Adam/simple_rnn_8/simple_rnn_cell_8/bias/v
<::Pd2,Adam/simple_rnn_9/simple_rnn_cell_9/kernel/v
F:Ddd26Adam/simple_rnn_9/simple_rnn_cell_9/recurrent_kernel/v
6:4d2*Adam/simple_rnn_9/simple_rnn_cell_9/bias/v?
#__inference__wrapped_model_11100961~ACBDFE23??<
5?2
0?-
simple_rnn_8_input?????????
? "1?.
,
dense_4!?
dense_4??????????
E__inference_dense_4_layer_call_and_return_conditional_losses_11103836\23/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? }
*__inference_dense_4_layer_call_fn_11103826O23/?,
%?"
 ?
inputs?????????d
? "???????????
G__inference_dropout_8_layer_call_and_return_conditional_losses_11103302d7?4
-?*
$?!
inputs?????????P
p 
? ")?&
?
0?????????P
? ?
G__inference_dropout_8_layer_call_and_return_conditional_losses_11103314d7?4
-?*
$?!
inputs?????????P
p
? ")?&
?
0?????????P
? ?
,__inference_dropout_8_layer_call_fn_11103292W7?4
-?*
$?!
inputs?????????P
p 
? "??????????P?
,__inference_dropout_8_layer_call_fn_11103297W7?4
-?*
$?!
inputs?????????P
p
? "??????????P?
G__inference_dropout_9_layer_call_and_return_conditional_losses_11103805\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? ?
G__inference_dropout_9_layer_call_and_return_conditional_losses_11103817\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? 
,__inference_dropout_9_layer_call_fn_11103795O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d
,__inference_dropout_9_layer_call_fn_11103800O3?0
)?&
 ?
inputs?????????d
p
? "??????????d?
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102261zACBDFE23G?D
=?:
0?-
simple_rnn_8_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102286zACBDFE23G?D
=?:
0?-
simple_rnn_8_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102554nACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_4_layer_call_and_return_conditional_losses_11102788nACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_4_layer_call_fn_11101832mACBDFE23G?D
=?:
0?-
simple_rnn_8_input?????????
p 

 
? "???????????
/__inference_sequential_4_layer_call_fn_11102236mACBDFE23G?D
=?:
0?-
simple_rnn_8_input?????????
p

 
? "???????????
/__inference_sequential_4_layer_call_fn_11102313aACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
/__inference_sequential_4_layer_call_fn_11102334aACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_11102811?ACBDFE23U?R
? 
K?H
F
simple_rnn_8_input0?-
simple_rnn_8_input?????????"1?.
,
dense_4!?
dense_4??????????
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11102963?ACBO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "2?/
(?%
0??????????????????P
? ?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11103071?ACBO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "2?/
(?%
0??????????????????P
? ?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11103179qACB??<
5?2
$?!
inputs?????????

 
p 

 
? ")?&
?
0?????????P
? ?
J__inference_simple_rnn_8_layer_call_and_return_conditional_losses_11103287qACB??<
5?2
$?!
inputs?????????

 
p

 
? ")?&
?
0?????????P
? ?
/__inference_simple_rnn_8_layer_call_fn_11102822}ACBO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"??????????????????P?
/__inference_simple_rnn_8_layer_call_fn_11102833}ACBO?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"??????????????????P?
/__inference_simple_rnn_8_layer_call_fn_11102844dACB??<
5?2
$?!
inputs?????????

 
p 

 
? "??????????P?
/__inference_simple_rnn_8_layer_call_fn_11102855dACB??<
5?2
$?!
inputs?????????

 
p

 
? "??????????P?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103466}DFEO?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p 

 
? "%?"
?
0?????????d
? ?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103574}DFEO?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p

 
? "%?"
?
0?????????d
? ?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103682mDFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "%?"
?
0?????????d
? ?
J__inference_simple_rnn_9_layer_call_and_return_conditional_losses_11103790mDFE??<
5?2
$?!
inputs?????????P

 
p

 
? "%?"
?
0?????????d
? ?
/__inference_simple_rnn_9_layer_call_fn_11103325pDFEO?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p 

 
? "??????????d?
/__inference_simple_rnn_9_layer_call_fn_11103336pDFEO?L
E?B
4?1
/?,
inputs/0??????????????????P

 
p

 
? "??????????d?
/__inference_simple_rnn_9_layer_call_fn_11103347`DFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "??????????d?
/__inference_simple_rnn_9_layer_call_fn_11103358`DFE??<
5?2
$?!
inputs?????????P

 
p

 
? "??????????d?
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11103881?ACB\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????P
p 
? "R?O
H?E
?
0/0?????????P
$?!
?
0/1/0?????????P
? ?
Q__inference_simple_rnn_cell_168_layer_call_and_return_conditional_losses_11103898?ACB\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????P
p
? "R?O
H?E
?
0/0?????????P
$?!
?
0/1/0?????????P
? ?
6__inference_simple_rnn_cell_168_layer_call_fn_11103850?ACB\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????P
p 
? "D?A
?
0?????????P
"?
?
1/0?????????P?
6__inference_simple_rnn_cell_168_layer_call_fn_11103864?ACB\?Y
R?O
 ?
inputs?????????
'?$
"?
states/0?????????P
p
? "D?A
?
0?????????P
"?
?
1/0?????????P?
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11103943?DFE\?Y
R?O
 ?
inputs?????????P
'?$
"?
states/0?????????d
p 
? "R?O
H?E
?
0/0?????????d
$?!
?
0/1/0?????????d
? ?
Q__inference_simple_rnn_cell_169_layer_call_and_return_conditional_losses_11103960?DFE\?Y
R?O
 ?
inputs?????????P
'?$
"?
states/0?????????d
p
? "R?O
H?E
?
0/0?????????d
$?!
?
0/1/0?????????d
? ?
6__inference_simple_rnn_cell_169_layer_call_fn_11103912?DFE\?Y
R?O
 ?
inputs?????????P
'?$
"?
states/0?????????d
p 
? "D?A
?
0?????????d
"?
?
1/0?????????d?
6__inference_simple_rnn_cell_169_layer_call_fn_11103926?DFE\?Y
R?O
 ?
inputs?????????P
'?$
"?
states/0?????????d
p
? "D?A
?
0?????????d
"?
?
1/0?????????d