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
?"serve*2.8.02v2.8.0-0-g3f878cff5b68??
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:d*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
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
'simple_rnn_14/simple_rnn_cell_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*8
shared_name)'simple_rnn_14/simple_rnn_cell_14/kernel
?
;simple_rnn_14/simple_rnn_cell_14/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_14/simple_rnn_cell_14/kernel*
_output_shapes

:P*
dtype0
?
1simple_rnn_14/simple_rnn_cell_14/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*B
shared_name31simple_rnn_14/simple_rnn_cell_14/recurrent_kernel
?
Esimple_rnn_14/simple_rnn_cell_14/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_14/simple_rnn_cell_14/recurrent_kernel*
_output_shapes

:PP*
dtype0
?
%simple_rnn_14/simple_rnn_cell_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*6
shared_name'%simple_rnn_14/simple_rnn_cell_14/bias
?
9simple_rnn_14/simple_rnn_cell_14/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_14/simple_rnn_cell_14/bias*
_output_shapes
:P*
dtype0
?
'simple_rnn_15/simple_rnn_cell_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*8
shared_name)'simple_rnn_15/simple_rnn_cell_15/kernel
?
;simple_rnn_15/simple_rnn_cell_15/kernel/Read/ReadVariableOpReadVariableOp'simple_rnn_15/simple_rnn_cell_15/kernel*
_output_shapes

:Pd*
dtype0
?
1simple_rnn_15/simple_rnn_cell_15/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*B
shared_name31simple_rnn_15/simple_rnn_cell_15/recurrent_kernel
?
Esimple_rnn_15/simple_rnn_cell_15/recurrent_kernel/Read/ReadVariableOpReadVariableOp1simple_rnn_15/simple_rnn_cell_15/recurrent_kernel*
_output_shapes

:dd*
dtype0
?
%simple_rnn_15/simple_rnn_cell_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*6
shared_name'%simple_rnn_15/simple_rnn_cell_15/bias
?
9simple_rnn_15/simple_rnn_cell_15/bias/Read/ReadVariableOpReadVariableOp%simple_rnn_15/simple_rnn_cell_15/bias*
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
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
?
.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*?
shared_name0.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/m
?
BAdam/simple_rnn_14/simple_rnn_cell_14/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/m*
_output_shapes

:P*
dtype0
?
8Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*I
shared_name:8Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/m
?
LAdam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/m*
_output_shapes

:PP*
dtype0
?
,Adam/simple_rnn_14/simple_rnn_cell_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*=
shared_name.,Adam/simple_rnn_14/simple_rnn_cell_14/bias/m
?
@Adam/simple_rnn_14/simple_rnn_cell_14/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_14/simple_rnn_cell_14/bias/m*
_output_shapes
:P*
dtype0
?
.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*?
shared_name0.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/m
?
BAdam/simple_rnn_15/simple_rnn_cell_15/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/m*
_output_shapes

:Pd*
dtype0
?
8Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*I
shared_name:8Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/m
?
LAdam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/m*
_output_shapes

:dd*
dtype0
?
,Adam/simple_rnn_15/simple_rnn_cell_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*=
shared_name.,Adam/simple_rnn_15/simple_rnn_cell_15/bias/m
?
@Adam/simple_rnn_15/simple_rnn_cell_15/bias/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_15/simple_rnn_cell_15/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0
?
.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*?
shared_name0.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/v
?
BAdam/simple_rnn_14/simple_rnn_cell_14/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/v*
_output_shapes

:P*
dtype0
?
8Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*I
shared_name:8Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/v
?
LAdam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/v*
_output_shapes

:PP*
dtype0
?
,Adam/simple_rnn_14/simple_rnn_cell_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*=
shared_name.,Adam/simple_rnn_14/simple_rnn_cell_14/bias/v
?
@Adam/simple_rnn_14/simple_rnn_cell_14/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_14/simple_rnn_cell_14/bias/v*
_output_shapes
:P*
dtype0
?
.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Pd*?
shared_name0.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/v
?
BAdam/simple_rnn_15/simple_rnn_cell_15/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/v*
_output_shapes

:Pd*
dtype0
?
8Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:dd*I
shared_name:8Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/v
?
LAdam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp8Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/v*
_output_shapes

:dd*
dtype0
?
,Adam/simple_rnn_15/simple_rnn_cell_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*=
shared_name.,Adam/simple_rnn_15/simple_rnn_cell_15/bias/v
?
@Adam/simple_rnn_15/simple_rnn_cell_15/bias/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_15/simple_rnn_cell_15/bias/v*
_output_shapes
:d*
dtype0

NoOpNoOp
?G
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
VARIABLE_VALUEdense_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ga
VARIABLE_VALUE'simple_rnn_14/simple_rnn_cell_14/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_14/simple_rnn_cell_14/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_14/simple_rnn_cell_14/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'simple_rnn_15/simple_rnn_cell_15/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1simple_rnn_15/simple_rnn_cell_15/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_15/simple_rnn_cell_15/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_14/simple_rnn_cell_14/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_15/simple_rnn_cell_15/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_14/simple_rnn_cell_14/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE8Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE,Adam/simple_rnn_15/simple_rnn_cell_15/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
#serving_default_simple_rnn_14_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall#serving_default_simple_rnn_14_input'simple_rnn_14/simple_rnn_cell_14/kernel%simple_rnn_14/simple_rnn_cell_14/bias1simple_rnn_14/simple_rnn_cell_14/recurrent_kernel'simple_rnn_15/simple_rnn_cell_15/kernel%simple_rnn_15/simple_rnn_cell_15/bias1simple_rnn_15/simple_rnn_cell_15/recurrent_kerneldense_7/kerneldense_7/bias*
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
&__inference_signature_wrapper_11352000
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp;simple_rnn_14/simple_rnn_cell_14/kernel/Read/ReadVariableOpEsimple_rnn_14/simple_rnn_cell_14/recurrent_kernel/Read/ReadVariableOp9simple_rnn_14/simple_rnn_cell_14/bias/Read/ReadVariableOp;simple_rnn_15/simple_rnn_cell_15/kernel/Read/ReadVariableOpEsimple_rnn_15/simple_rnn_cell_15/recurrent_kernel/Read/ReadVariableOp9simple_rnn_15/simple_rnn_cell_15/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOpBAdam/simple_rnn_14/simple_rnn_cell_14/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_14/simple_rnn_cell_14/bias/m/Read/ReadVariableOpBAdam/simple_rnn_15/simple_rnn_cell_15/kernel/m/Read/ReadVariableOpLAdam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/m/Read/ReadVariableOp@Adam/simple_rnn_15/simple_rnn_cell_15/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpBAdam/simple_rnn_14/simple_rnn_cell_14/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_14/simple_rnn_cell_14/bias/v/Read/ReadVariableOpBAdam/simple_rnn_15/simple_rnn_cell_15/kernel/v/Read/ReadVariableOpLAdam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/v/Read/ReadVariableOp@Adam/simple_rnn_15/simple_rnn_cell_15/bias/v/Read/ReadVariableOpConst*,
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
!__inference__traced_save_11353265
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate'simple_rnn_14/simple_rnn_cell_14/kernel1simple_rnn_14/simple_rnn_cell_14/recurrent_kernel%simple_rnn_14/simple_rnn_cell_14/bias'simple_rnn_15/simple_rnn_cell_15/kernel1simple_rnn_15/simple_rnn_cell_15/recurrent_kernel%simple_rnn_15/simple_rnn_cell_15/biastotalcountAdam/dense_7/kernel/mAdam/dense_7/bias/m.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/m8Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/m,Adam/simple_rnn_14/simple_rnn_cell_14/bias/m.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/m8Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/m,Adam/simple_rnn_15/simple_rnn_cell_15/bias/mAdam/dense_7/kernel/vAdam/dense_7/bias/v.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/v8Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/v,Adam/simple_rnn_14/simple_rnn_cell_14/bias/v.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/v8Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/v,Adam/simple_rnn_15/simple_rnn_cell_15/bias/v*+
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
$__inference__traced_restore_11353368??
?
I
-__inference_dropout_15_layer_call_fn_11352984

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_11350983`
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
?
?
0__inference_simple_rnn_14_layer_call_fn_11352022
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11350433|
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
while_body_11352086
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_174_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_174_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_174_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_174/MatMul/ReadVariableOp?1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_174_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_174/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_174/BiasAddBiasAdd*while/simple_rnn_cell_174/MatMul:product:08while/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_174/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_174/addAddV2*while/simple_rnn_cell_174/BiasAdd:output:0,while/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_174/TanhTanh!while/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_174/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_174/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_174/MatMul/ReadVariableOp2^while/simple_rnn_cell_174/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_174_biasadd_readvariableop_resource;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_174_matmul_readvariableop_resource:while_simple_rnn_cell_174_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_174/MatMul/ReadVariableOp/while/simple_rnn_cell_174/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp: 
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
?!
?
while_body_11350503
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_175_11350525_0:Pd2
$while_simple_rnn_cell_175_11350527_0:d6
$while_simple_rnn_cell_175_11350529_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_175_11350525:Pd0
"while_simple_rnn_cell_175_11350527:d4
"while_simple_rnn_cell_175_11350529:dd??1while/simple_rnn_cell_175/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_175/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_175_11350525_0$while_simple_rnn_cell_175_11350527_0$while_simple_rnn_cell_175_11350529_0*
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
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11350490?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_175/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_175/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_175/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_175_11350525$while_simple_rnn_cell_175_11350525_0"J
"while_simple_rnn_cell_175_11350527$while_simple_rnn_cell_175_11350527_0"J
"while_simple_rnn_cell_175_11350529$while_simple_rnn_cell_175_11350529_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_175/StatefulPartitionedCall1while/simple_rnn_cell_175/StatefulPartitionedCall: 
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
?>
?
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352655
inputs_0D
2simple_rnn_cell_175_matmul_readvariableop_resource:PdA
3simple_rnn_cell_175_biasadd_readvariableop_resource:dF
4simple_rnn_cell_175_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_175/BiasAdd/ReadVariableOp?)simple_rnn_cell_175/MatMul/ReadVariableOp?+simple_rnn_cell_175/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_175_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_175/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_175_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_175/BiasAddBiasAdd$simple_rnn_cell_175/MatMul:product:02simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_175_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_175/MatMul_1MatMulzeros:output:03simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_175/addAddV2$simple_rnn_cell_175/BiasAdd:output:0&simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_175/TanhTanhsimple_rnn_cell_175/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_175_matmul_readvariableop_resource3simple_rnn_cell_175_biasadd_readvariableop_resource4simple_rnn_cell_175_matmul_1_readvariableop_resource*
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
while_body_11352589*
condR
while_cond_11352588*8
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
NoOpNoOp+^simple_rnn_cell_175/BiasAdd/ReadVariableOp*^simple_rnn_cell_175/MatMul/ReadVariableOp,^simple_rnn_cell_175/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_175/BiasAdd/ReadVariableOp*simple_rnn_cell_175/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_175/MatMul/ReadVariableOp)simple_rnn_cell_175/MatMul/ReadVariableOp2Z
+simple_rnn_cell_175/MatMul_1/ReadVariableOp+simple_rnn_cell_175/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?
?
*__inference_dense_7_layer_call_fn_11353015

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
E__inference_dense_7_layer_call_and_return_conditional_losses_11350995o
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
?
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351475
simple_rnn_14_input(
simple_rnn_14_11351453:P$
simple_rnn_14_11351455:P(
simple_rnn_14_11351457:PP(
simple_rnn_15_11351461:Pd$
simple_rnn_15_11351463:d(
simple_rnn_15_11351465:dd"
dense_7_11351469:d
dense_7_11351471:
identity??dense_7/StatefulPartitionedCall?"dropout_14/StatefulPartitionedCall?"dropout_15/StatefulPartitionedCall?%simple_rnn_14/StatefulPartitionedCall?%simple_rnn_15/StatefulPartitionedCall?
%simple_rnn_14/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_14_inputsimple_rnn_14_11351453simple_rnn_14_11351455simple_rnn_14_11351457*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11351328?
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_11351204?
%simple_rnn_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0simple_rnn_15_11351461simple_rnn_15_11351463simple_rnn_15_11351465*
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11351175?
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_15/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_11351051?
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_7_11351469dense_7_11351471*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_11350995w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_7/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall&^simple_rnn_14/StatefulPartitionedCall&^simple_rnn_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2N
%simple_rnn_14/StatefulPartitionedCall%simple_rnn_14/StatefulPartitionedCall2N
%simple_rnn_15/StatefulPartitionedCall%simple_rnn_15/StatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_14_input
?4
?
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11350725

inputs.
simple_rnn_cell_175_11350650:Pd*
simple_rnn_cell_175_11350652:d.
simple_rnn_cell_175_11350654:dd
identity??+simple_rnn_cell_175/StatefulPartitionedCall?while;
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
+simple_rnn_cell_175/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_175_11350650simple_rnn_cell_175_11350652simple_rnn_cell_175_11350654*
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
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11350610n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_175_11350650simple_rnn_cell_175_11350652simple_rnn_cell_175_11350654*
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
while_body_11350662*
condR
while_cond_11350661*8
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
NoOpNoOp,^simple_rnn_cell_175/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_175/StatefulPartitionedCall+simple_rnn_cell_175/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?
?
while_cond_11352804
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11352804___redundant_placeholder06
2while_while_cond_11352804___redundant_placeholder16
2while_while_cond_11352804___redundant_placeholder26
2while_while_cond_11352804___redundant_placeholder3
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
while_body_11351262
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_174_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_174_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_174_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_174/MatMul/ReadVariableOp?1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_174_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_174/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_174/BiasAddBiasAdd*while/simple_rnn_cell_174/MatMul:product:08while/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_174/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_174/addAddV2*while/simple_rnn_cell_174/BiasAdd:output:0,while/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_174/TanhTanh!while/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_174/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_174/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_174/MatMul/ReadVariableOp2^while/simple_rnn_cell_174/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_174_biasadd_readvariableop_resource;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_174_matmul_readvariableop_resource:while_simple_rnn_cell_174_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_174/MatMul/ReadVariableOp/while/simple_rnn_cell_174/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp: 
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
?!
?
while_body_11350211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_174_11350233_0:P2
$while_simple_rnn_cell_174_11350235_0:P6
$while_simple_rnn_cell_174_11350237_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_174_11350233:P0
"while_simple_rnn_cell_174_11350235:P4
"while_simple_rnn_cell_174_11350237:PP??1while/simple_rnn_cell_174/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_174/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_174_11350233_0$while_simple_rnn_cell_174_11350235_0$while_simple_rnn_cell_174_11350237_0*
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
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11350198?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_174/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_174/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_174/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_174_11350233$while_simple_rnn_cell_174_11350233_0"J
"while_simple_rnn_cell_174_11350235$while_simple_rnn_cell_174_11350235_0"J
"while_simple_rnn_cell_174_11350237$while_simple_rnn_cell_174_11350237_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_174/StatefulPartitionedCall1while/simple_rnn_cell_174/StatefulPartitionedCall: 
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
?>
?
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352260
inputs_0D
2simple_rnn_cell_174_matmul_readvariableop_resource:PA
3simple_rnn_cell_174_biasadd_readvariableop_resource:PF
4simple_rnn_cell_174_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_174/BiasAdd/ReadVariableOp?)simple_rnn_cell_174/MatMul/ReadVariableOp?+simple_rnn_cell_174/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_174_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_174/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_174_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_174/BiasAddBiasAdd$simple_rnn_cell_174/MatMul:product:02simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_174_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_174/MatMul_1MatMulzeros:output:03simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_174/addAddV2$simple_rnn_cell_174/BiasAdd:output:0&simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_174/TanhTanhsimple_rnn_cell_174/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_174_matmul_readvariableop_resource3simple_rnn_cell_174_biasadd_readvariableop_resource4simple_rnn_cell_174_matmul_1_readvariableop_resource*
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
while_body_11352194*
condR
while_cond_11352193*8
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
NoOpNoOp+^simple_rnn_cell_174/BiasAdd/ReadVariableOp*^simple_rnn_cell_174/MatMul/ReadVariableOp,^simple_rnn_cell_174/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_174/BiasAdd/ReadVariableOp*simple_rnn_cell_174/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_174/MatMul/ReadVariableOp)simple_rnn_cell_174/MatMul/ReadVariableOp2Z
+simple_rnn_cell_174/MatMul_1/ReadVariableOp+simple_rnn_cell_174/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?	
g
H__inference_dropout_15_layer_call_and_return_conditional_losses_11351051

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
?

?
!simple_rnn_14_while_cond_113515648
4simple_rnn_14_while_simple_rnn_14_while_loop_counter>
:simple_rnn_14_while_simple_rnn_14_while_maximum_iterations#
simple_rnn_14_while_placeholder%
!simple_rnn_14_while_placeholder_1%
!simple_rnn_14_while_placeholder_2:
6simple_rnn_14_while_less_simple_rnn_14_strided_slice_1R
Nsimple_rnn_14_while_simple_rnn_14_while_cond_11351564___redundant_placeholder0R
Nsimple_rnn_14_while_simple_rnn_14_while_cond_11351564___redundant_placeholder1R
Nsimple_rnn_14_while_simple_rnn_14_while_cond_11351564___redundant_placeholder2R
Nsimple_rnn_14_while_simple_rnn_14_while_cond_11351564___redundant_placeholder3 
simple_rnn_14_while_identity
?
simple_rnn_14/while/LessLesssimple_rnn_14_while_placeholder6simple_rnn_14_while_less_simple_rnn_14_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_14/while/IdentityIdentitysimple_rnn_14/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_14_while_identity%simple_rnn_14/while/Identity:output:0*(
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
?
/__inference_sequential_7_layer_call_fn_11351502

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
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351002o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
6__inference_simple_rnn_cell_175_layer_call_fn_11353101

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
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11350490o
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
ق
?
$__inference__traced_restore_11353368
file_prefix1
assignvariableop_dense_7_kernel:d-
assignvariableop_1_dense_7_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: L
:assignvariableop_7_simple_rnn_14_simple_rnn_cell_14_kernel:PV
Dassignvariableop_8_simple_rnn_14_simple_rnn_cell_14_recurrent_kernel:PPF
8assignvariableop_9_simple_rnn_14_simple_rnn_cell_14_bias:PM
;assignvariableop_10_simple_rnn_15_simple_rnn_cell_15_kernel:PdW
Eassignvariableop_11_simple_rnn_15_simple_rnn_cell_15_recurrent_kernel:ddG
9assignvariableop_12_simple_rnn_15_simple_rnn_cell_15_bias:d#
assignvariableop_13_total: #
assignvariableop_14_count: ;
)assignvariableop_15_adam_dense_7_kernel_m:d5
'assignvariableop_16_adam_dense_7_bias_m:T
Bassignvariableop_17_adam_simple_rnn_14_simple_rnn_cell_14_kernel_m:P^
Lassignvariableop_18_adam_simple_rnn_14_simple_rnn_cell_14_recurrent_kernel_m:PPN
@assignvariableop_19_adam_simple_rnn_14_simple_rnn_cell_14_bias_m:PT
Bassignvariableop_20_adam_simple_rnn_15_simple_rnn_cell_15_kernel_m:Pd^
Lassignvariableop_21_adam_simple_rnn_15_simple_rnn_cell_15_recurrent_kernel_m:ddN
@assignvariableop_22_adam_simple_rnn_15_simple_rnn_cell_15_bias_m:d;
)assignvariableop_23_adam_dense_7_kernel_v:d5
'assignvariableop_24_adam_dense_7_bias_v:T
Bassignvariableop_25_adam_simple_rnn_14_simple_rnn_cell_14_kernel_v:P^
Lassignvariableop_26_adam_simple_rnn_14_simple_rnn_cell_14_recurrent_kernel_v:PPN
@assignvariableop_27_adam_simple_rnn_14_simple_rnn_cell_14_bias_v:PT
Bassignvariableop_28_adam_simple_rnn_15_simple_rnn_cell_15_kernel_v:Pd^
Lassignvariableop_29_adam_simple_rnn_15_simple_rnn_cell_15_recurrent_kernel_v:ddN
@assignvariableop_30_adam_simple_rnn_15_simple_rnn_cell_15_bias_v:d
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
AssignVariableOpAssignVariableOpassignvariableop_dense_7_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_7_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp:assignvariableop_7_simple_rnn_14_simple_rnn_cell_14_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpDassignvariableop_8_simple_rnn_14_simple_rnn_cell_14_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp8assignvariableop_9_simple_rnn_14_simple_rnn_cell_14_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp;assignvariableop_10_simple_rnn_15_simple_rnn_cell_15_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpEassignvariableop_11_simple_rnn_15_simple_rnn_cell_15_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp9assignvariableop_12_simple_rnn_15_simple_rnn_cell_15_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_15AssignVariableOp)assignvariableop_15_adam_dense_7_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_adam_dense_7_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpBassignvariableop_17_adam_simple_rnn_14_simple_rnn_cell_14_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpLassignvariableop_18_adam_simple_rnn_14_simple_rnn_cell_14_recurrent_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp@assignvariableop_19_adam_simple_rnn_14_simple_rnn_cell_14_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpBassignvariableop_20_adam_simple_rnn_15_simple_rnn_cell_15_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpLassignvariableop_21_adam_simple_rnn_15_simple_rnn_cell_15_recurrent_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp@assignvariableop_22_adam_simple_rnn_15_simple_rnn_cell_15_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_7_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_7_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpBassignvariableop_25_adam_simple_rnn_14_simple_rnn_cell_14_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpLassignvariableop_26_adam_simple_rnn_14_simple_rnn_cell_14_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_simple_rnn_14_simple_rnn_cell_14_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpBassignvariableop_28_adam_simple_rnn_15_simple_rnn_cell_15_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpLassignvariableop_29_adam_simple_rnn_15_simple_rnn_cell_15_recurrent_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_simple_rnn_15_simple_rnn_cell_15_bias_vIdentity_30:output:0"/device:CPU:0*
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
̫
?	
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351977

inputsR
@simple_rnn_14_simple_rnn_cell_174_matmul_readvariableop_resource:PO
Asimple_rnn_14_simple_rnn_cell_174_biasadd_readvariableop_resource:PT
Bsimple_rnn_14_simple_rnn_cell_174_matmul_1_readvariableop_resource:PPR
@simple_rnn_15_simple_rnn_cell_175_matmul_readvariableop_resource:PdO
Asimple_rnn_15_simple_rnn_cell_175_biasadd_readvariableop_resource:dT
Bsimple_rnn_15_simple_rnn_cell_175_matmul_1_readvariableop_resource:dd8
&dense_7_matmul_readvariableop_resource:d5
'dense_7_biasadd_readvariableop_resource:
identity??dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?8simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp?7simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp?9simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp?simple_rnn_14/while?8simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp?7simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp?9simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp?simple_rnn_15/whileI
simple_rnn_14/ShapeShapeinputs*
T0*
_output_shapes
:k
!simple_rnn_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_14/strided_sliceStridedSlicesimple_rnn_14/Shape:output:0*simple_rnn_14/strided_slice/stack:output:0,simple_rnn_14/strided_slice/stack_1:output:0,simple_rnn_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_14/zeros/packedPack$simple_rnn_14/strided_slice:output:0%simple_rnn_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_14/zerosFill#simple_rnn_14/zeros/packed:output:0"simple_rnn_14/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pq
simple_rnn_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_14/transpose	Transposeinputs%simple_rnn_14/transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
simple_rnn_14/Shape_1Shapesimple_rnn_14/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_14/strided_slice_1StridedSlicesimple_rnn_14/Shape_1:output:0,simple_rnn_14/strided_slice_1/stack:output:0.simple_rnn_14/strided_slice_1/stack_1:output:0.simple_rnn_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_14/TensorArrayV2TensorListReserve2simple_rnn_14/TensorArrayV2/element_shape:output:0&simple_rnn_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
5simple_rnn_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_14/transpose:y:0Lsimple_rnn_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_14/strided_slice_2StridedSlicesimple_rnn_14/transpose:y:0,simple_rnn_14/strided_slice_2/stack:output:0.simple_rnn_14/strided_slice_2/stack_1:output:0.simple_rnn_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
7simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp@simple_rnn_14_simple_rnn_cell_174_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
(simple_rnn_14/simple_rnn_cell_174/MatMulMatMul&simple_rnn_14/strided_slice_2:output:0?simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_14_simple_rnn_cell_174_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
)simple_rnn_14/simple_rnn_cell_174/BiasAddBiasAdd2simple_rnn_14/simple_rnn_cell_174/MatMul:product:0@simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
9simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_14_simple_rnn_cell_174_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
*simple_rnn_14/simple_rnn_cell_174/MatMul_1MatMulsimple_rnn_14/zeros:output:0Asimple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_14/simple_rnn_cell_174/addAddV22simple_rnn_14/simple_rnn_cell_174/BiasAdd:output:04simple_rnn_14/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
&simple_rnn_14/simple_rnn_cell_174/TanhTanh)simple_rnn_14/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P|
+simple_rnn_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_14/TensorArrayV2_1TensorListReserve4simple_rnn_14/TensorArrayV2_1/element_shape:output:0&simple_rnn_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_14/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_14/whileWhile)simple_rnn_14/while/loop_counter:output:0/simple_rnn_14/while/maximum_iterations:output:0simple_rnn_14/time:output:0&simple_rnn_14/TensorArrayV2_1:handle:0simple_rnn_14/zeros:output:0&simple_rnn_14/strided_slice_1:output:0Esimple_rnn_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_14_simple_rnn_cell_174_matmul_readvariableop_resourceAsimple_rnn_14_simple_rnn_cell_174_biasadd_readvariableop_resourceBsimple_rnn_14_simple_rnn_cell_174_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *-
body%R#
!simple_rnn_14_while_body_11351785*-
cond%R#
!simple_rnn_14_while_cond_11351784*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
>simple_rnn_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
0simple_rnn_14/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_14/while:output:3Gsimple_rnn_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0v
#simple_rnn_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_14/strided_slice_3StridedSlice9simple_rnn_14/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_14/strided_slice_3/stack:output:0.simple_rnn_14/strided_slice_3/stack_1:output:0.simple_rnn_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_masks
simple_rnn_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_14/transpose_1	Transpose9simple_rnn_14/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_14/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P]
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_14/dropout/MulMulsimple_rnn_14/transpose_1:y:0!dropout_14/dropout/Const:output:0*
T0*+
_output_shapes
:?????????Pe
dropout_14/dropout/ShapeShapesimple_rnn_14/transpose_1:y:0*
T0*
_output_shapes
:?
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
dtype0f
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????P?
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????P?
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P_
simple_rnn_15/ShapeShapedropout_14/dropout/Mul_1:z:0*
T0*
_output_shapes
:k
!simple_rnn_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_15/strided_sliceStridedSlicesimple_rnn_15/Shape:output:0*simple_rnn_15/strided_slice/stack:output:0,simple_rnn_15/strided_slice/stack_1:output:0,simple_rnn_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_15/zeros/packedPack$simple_rnn_15/strided_slice:output:0%simple_rnn_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_15/zerosFill#simple_rnn_15/zeros/packed:output:0"simple_rnn_15/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dq
simple_rnn_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_15/transpose	Transposedropout_14/dropout/Mul_1:z:0%simple_rnn_15/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P`
simple_rnn_15/Shape_1Shapesimple_rnn_15/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_15/strided_slice_1StridedSlicesimple_rnn_15/Shape_1:output:0,simple_rnn_15/strided_slice_1/stack:output:0.simple_rnn_15/strided_slice_1/stack_1:output:0.simple_rnn_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_15/TensorArrayV2TensorListReserve2simple_rnn_15/TensorArrayV2/element_shape:output:0&simple_rnn_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
5simple_rnn_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_15/transpose:y:0Lsimple_rnn_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_15/strided_slice_2StridedSlicesimple_rnn_15/transpose:y:0,simple_rnn_15/strided_slice_2/stack:output:0.simple_rnn_15/strided_slice_2/stack_1:output:0.simple_rnn_15/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
7simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp@simple_rnn_15_simple_rnn_cell_175_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
(simple_rnn_15/simple_rnn_cell_175/MatMulMatMul&simple_rnn_15/strided_slice_2:output:0?simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_15_simple_rnn_cell_175_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
)simple_rnn_15/simple_rnn_cell_175/BiasAddBiasAdd2simple_rnn_15/simple_rnn_cell_175/MatMul:product:0@simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
9simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_15_simple_rnn_cell_175_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
*simple_rnn_15/simple_rnn_cell_175/MatMul_1MatMulsimple_rnn_15/zeros:output:0Asimple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_15/simple_rnn_cell_175/addAddV22simple_rnn_15/simple_rnn_cell_175/BiasAdd:output:04simple_rnn_15/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
&simple_rnn_15/simple_rnn_cell_175/TanhTanh)simple_rnn_15/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d|
+simple_rnn_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_15/TensorArrayV2_1TensorListReserve4simple_rnn_15/TensorArrayV2_1/element_shape:output:0&simple_rnn_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_15/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_15/whileWhile)simple_rnn_15/while/loop_counter:output:0/simple_rnn_15/while/maximum_iterations:output:0simple_rnn_15/time:output:0&simple_rnn_15/TensorArrayV2_1:handle:0simple_rnn_15/zeros:output:0&simple_rnn_15/strided_slice_1:output:0Esimple_rnn_15/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_15_simple_rnn_cell_175_matmul_readvariableop_resourceAsimple_rnn_15_simple_rnn_cell_175_biasadd_readvariableop_resourceBsimple_rnn_15_simple_rnn_cell_175_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *-
body%R#
!simple_rnn_15_while_body_11351897*-
cond%R#
!simple_rnn_15_while_cond_11351896*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
>simple_rnn_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
0simple_rnn_15/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_15/while:output:3Gsimple_rnn_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0v
#simple_rnn_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_15/strided_slice_3StridedSlice9simple_rnn_15/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_15/strided_slice_3/stack:output:0.simple_rnn_15/strided_slice_3/stack_1:output:0.simple_rnn_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_masks
simple_rnn_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_15/transpose_1	Transpose9simple_rnn_15/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d]
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_15/dropout/MulMul&simple_rnn_15/strided_slice_3:output:0!dropout_15/dropout/Const:output:0*
T0*'
_output_shapes
:?????????dn
dropout_15/dropout/ShapeShape&simple_rnn_15/strided_slice_3:output:0*
T0*
_output_shapes
:?
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*'
_output_shapes
:?????????d*
dtype0f
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????d?
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????d?
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????d?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_7/MatMulMatMuldropout_15/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp9^simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp8^simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp:^simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp^simple_rnn_14/while9^simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp8^simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp:^simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp^simple_rnn_15/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2t
8simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp8simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp2r
7simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp7simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp2v
9simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp9simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp2*
simple_rnn_14/whilesimple_rnn_14/while2t
8simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp8simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp2r
7simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp7simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp2v
9simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp9simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp2*
simple_rnn_15/whilesimple_rnn_15/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351002

inputs(
simple_rnn_14_11350849:P$
simple_rnn_14_11350851:P(
simple_rnn_14_11350853:PP(
simple_rnn_15_11350971:Pd$
simple_rnn_15_11350973:d(
simple_rnn_15_11350975:dd"
dense_7_11350996:d
dense_7_11350998:
identity??dense_7/StatefulPartitionedCall?%simple_rnn_14/StatefulPartitionedCall?%simple_rnn_15/StatefulPartitionedCall?
%simple_rnn_14/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_14_11350849simple_rnn_14_11350851simple_rnn_14_11350853*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11350848?
dropout_14/PartitionedCallPartitionedCall.simple_rnn_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_11350861?
%simple_rnn_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0simple_rnn_15_11350971simple_rnn_15_11350973simple_rnn_15_11350975*
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11350970?
dropout_15/PartitionedCallPartitionedCall.simple_rnn_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_11350983?
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_7_11350996dense_7_11350998*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_11350995w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_7/StatefulPartitionedCall&^simple_rnn_14/StatefulPartitionedCall&^simple_rnn_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2N
%simple_rnn_14/StatefulPartitionedCall%simple_rnn_14/StatefulPartitionedCall2N
%simple_rnn_15/StatefulPartitionedCall%simple_rnn_15/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
!simple_rnn_14_while_cond_113517848
4simple_rnn_14_while_simple_rnn_14_while_loop_counter>
:simple_rnn_14_while_simple_rnn_14_while_maximum_iterations#
simple_rnn_14_while_placeholder%
!simple_rnn_14_while_placeholder_1%
!simple_rnn_14_while_placeholder_2:
6simple_rnn_14_while_less_simple_rnn_14_strided_slice_1R
Nsimple_rnn_14_while_simple_rnn_14_while_cond_11351784___redundant_placeholder0R
Nsimple_rnn_14_while_simple_rnn_14_while_cond_11351784___redundant_placeholder1R
Nsimple_rnn_14_while_simple_rnn_14_while_cond_11351784___redundant_placeholder2R
Nsimple_rnn_14_while_simple_rnn_14_while_cond_11351784___redundant_placeholder3 
simple_rnn_14_while_identity
?
simple_rnn_14/while/LessLesssimple_rnn_14_while_placeholder6simple_rnn_14_while_less_simple_rnn_14_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_14/while/IdentityIdentitysimple_rnn_14/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_14_while_identity%simple_rnn_14/while/Identity:output:0*(
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
while_body_11352302
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_174_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_174_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_174_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_174/MatMul/ReadVariableOp?1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_174_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_174/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_174/BiasAddBiasAdd*while/simple_rnn_cell_174/MatMul:product:08while/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_174/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_174/addAddV2*while/simple_rnn_cell_174/BiasAdd:output:0,while/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_174/TanhTanh!while/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_174/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_174/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_174/MatMul/ReadVariableOp2^while/simple_rnn_cell_174/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_174_biasadd_readvariableop_resource;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_174_matmul_readvariableop_resource:while_simple_rnn_cell_174_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_174/MatMul/ReadVariableOp/while/simple_rnn_cell_174/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp: 
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
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11350610

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
?

g
H__inference_dropout_14_layer_call_and_return_conditional_losses_11351204

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
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
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
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
while_cond_11351108
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11351108___redundant_placeholder06
2while_while_cond_11351108___redundant_placeholder16
2while_while_cond_11351108___redundant_placeholder26
2while_while_cond_11351108___redundant_placeholder3
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
while_cond_11352301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11352301___redundant_placeholder06
2while_while_cond_11352301___redundant_placeholder16
2while_while_cond_11352301___redundant_placeholder26
2while_while_cond_11352301___redundant_placeholder3
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
?
I
-__inference_dropout_14_layer_call_fn_11352481

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
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_11350861d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?F
?
.sequential_7_simple_rnn_15_while_body_11350077R
Nsequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_while_loop_counterX
Tsequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_while_maximum_iterations0
,sequential_7_simple_rnn_15_while_placeholder2
.sequential_7_simple_rnn_15_while_placeholder_12
.sequential_7_simple_rnn_15_while_placeholder_2Q
Msequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_strided_slice_1_0?
?sequential_7_simple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_15_tensorarrayunstack_tensorlistfromtensor_0g
Usequential_7_simple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource_0:Pdd
Vsequential_7_simple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource_0:di
Wsequential_7_simple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0:dd-
)sequential_7_simple_rnn_15_while_identity/
+sequential_7_simple_rnn_15_while_identity_1/
+sequential_7_simple_rnn_15_while_identity_2/
+sequential_7_simple_rnn_15_while_identity_3/
+sequential_7_simple_rnn_15_while_identity_4O
Ksequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_strided_slice_1?
?sequential_7_simple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_15_tensorarrayunstack_tensorlistfromtensore
Ssequential_7_simple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource:Pdb
Tsequential_7_simple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource:dg
Usequential_7_simple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource:dd??Ksequential_7/simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp?Jsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp?Lsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp?
Rsequential_7/simple_rnn_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Dsequential_7/simple_rnn_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_7_simple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_15_tensorarrayunstack_tensorlistfromtensor_0,sequential_7_simple_rnn_15_while_placeholder[sequential_7/simple_rnn_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
Jsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOpUsequential_7_simple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
;sequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMulMatMulKsequential_7/simple_rnn_15/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Ksequential_7/simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOpVsequential_7_simple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
<sequential_7/simple_rnn_15/while/simple_rnn_cell_175/BiasAddBiasAddEsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul:product:0Ssequential_7/simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Lsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOpWsequential_7_simple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
=sequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul_1MatMul.sequential_7_simple_rnn_15_while_placeholder_2Tsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8sequential_7/simple_rnn_15/while/simple_rnn_cell_175/addAddV2Esequential_7/simple_rnn_15/while/simple_rnn_cell_175/BiasAdd:output:0Gsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
9sequential_7/simple_rnn_15/while/simple_rnn_cell_175/TanhTanh<sequential_7/simple_rnn_15/while/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d?
Esequential_7/simple_rnn_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_7_simple_rnn_15_while_placeholder_1,sequential_7_simple_rnn_15_while_placeholder=sequential_7/simple_rnn_15/while/simple_rnn_cell_175/Tanh:y:0*
_output_shapes
: *
element_dtype0:???h
&sequential_7/simple_rnn_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
$sequential_7/simple_rnn_15/while/addAddV2,sequential_7_simple_rnn_15_while_placeholder/sequential_7/simple_rnn_15/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_7/simple_rnn_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
&sequential_7/simple_rnn_15/while/add_1AddV2Nsequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_while_loop_counter1sequential_7/simple_rnn_15/while/add_1/y:output:0*
T0*
_output_shapes
: ?
)sequential_7/simple_rnn_15/while/IdentityIdentity*sequential_7/simple_rnn_15/while/add_1:z:0&^sequential_7/simple_rnn_15/while/NoOp*
T0*
_output_shapes
: ?
+sequential_7/simple_rnn_15/while/Identity_1IdentityTsequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_while_maximum_iterations&^sequential_7/simple_rnn_15/while/NoOp*
T0*
_output_shapes
: ?
+sequential_7/simple_rnn_15/while/Identity_2Identity(sequential_7/simple_rnn_15/while/add:z:0&^sequential_7/simple_rnn_15/while/NoOp*
T0*
_output_shapes
: ?
+sequential_7/simple_rnn_15/while/Identity_3IdentityUsequential_7/simple_rnn_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_7/simple_rnn_15/while/NoOp*
T0*
_output_shapes
: :????
+sequential_7/simple_rnn_15/while/Identity_4Identity=sequential_7/simple_rnn_15/while/simple_rnn_cell_175/Tanh:y:0&^sequential_7/simple_rnn_15/while/NoOp*
T0*'
_output_shapes
:?????????d?
%sequential_7/simple_rnn_15/while/NoOpNoOpL^sequential_7/simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOpK^sequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOpM^sequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_7_simple_rnn_15_while_identity2sequential_7/simple_rnn_15/while/Identity:output:0"c
+sequential_7_simple_rnn_15_while_identity_14sequential_7/simple_rnn_15/while/Identity_1:output:0"c
+sequential_7_simple_rnn_15_while_identity_24sequential_7/simple_rnn_15/while/Identity_2:output:0"c
+sequential_7_simple_rnn_15_while_identity_34sequential_7/simple_rnn_15/while/Identity_3:output:0"c
+sequential_7_simple_rnn_15_while_identity_44sequential_7/simple_rnn_15/while/Identity_4:output:0"?
Ksequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_strided_slice_1Msequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_strided_slice_1_0"?
Tsequential_7_simple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resourceVsequential_7_simple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource_0"?
Usequential_7_simple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resourceWsequential_7_simple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0"?
Ssequential_7_simple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resourceUsequential_7_simple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource_0"?
?sequential_7_simple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_15_tensorarrayunstack_tensorlistfromtensor?sequential_7_simple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
Ksequential_7/simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOpKsequential_7/simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp2?
Jsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOpJsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp2?
Lsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOpLsequential_7/simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp: 
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
while_body_11352194
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_174_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_174_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_174_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_174/MatMul/ReadVariableOp?1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_174_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_174/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_174/BiasAddBiasAdd*while/simple_rnn_cell_174/MatMul:product:08while/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_174/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_174/addAddV2*while/simple_rnn_cell_174/BiasAdd:output:0,while/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_174/TanhTanh!while/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_174/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_174/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_174/MatMul/ReadVariableOp2^while/simple_rnn_cell_174/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_174_biasadd_readvariableop_resource;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_174_matmul_readvariableop_resource:while_simple_rnn_cell_174_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_174/MatMul/ReadVariableOp/while/simple_rnn_cell_174/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp: 
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
?
E__inference_dense_7_layer_call_and_return_conditional_losses_11350995

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
0__inference_simple_rnn_15_layer_call_fn_11352547

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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11351175o
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
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?	
g
H__inference_dropout_15_layer_call_and_return_conditional_losses_11353006

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
?
?
.sequential_7_simple_rnn_14_while_cond_11349971R
Nsequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_while_loop_counterX
Tsequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_while_maximum_iterations0
,sequential_7_simple_rnn_14_while_placeholder2
.sequential_7_simple_rnn_14_while_placeholder_12
.sequential_7_simple_rnn_14_while_placeholder_2T
Psequential_7_simple_rnn_14_while_less_sequential_7_simple_rnn_14_strided_slice_1l
hsequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_while_cond_11349971___redundant_placeholder0l
hsequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_while_cond_11349971___redundant_placeholder1l
hsequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_while_cond_11349971___redundant_placeholder2l
hsequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_while_cond_11349971___redundant_placeholder3-
)sequential_7_simple_rnn_14_while_identity
?
%sequential_7/simple_rnn_14/while/LessLess,sequential_7_simple_rnn_14_while_placeholderPsequential_7_simple_rnn_14_while_less_sequential_7_simple_rnn_14_strided_slice_1*
T0*
_output_shapes
: ?
)sequential_7/simple_rnn_14/while/IdentityIdentity)sequential_7/simple_rnn_14/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_7_simple_rnn_14_while_identity2sequential_7/simple_rnn_14/while/Identity:output:0*(
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
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352763
inputs_0D
2simple_rnn_cell_175_matmul_readvariableop_resource:PdA
3simple_rnn_cell_175_biasadd_readvariableop_resource:dF
4simple_rnn_cell_175_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_175/BiasAdd/ReadVariableOp?)simple_rnn_cell_175/MatMul/ReadVariableOp?+simple_rnn_cell_175/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_175_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_175/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_175_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_175/BiasAddBiasAdd$simple_rnn_cell_175/MatMul:product:02simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_175_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_175/MatMul_1MatMulzeros:output:03simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_175/addAddV2$simple_rnn_cell_175/BiasAdd:output:0&simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_175/TanhTanhsimple_rnn_cell_175/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_175_matmul_readvariableop_resource3simple_rnn_cell_175_biasadd_readvariableop_resource4simple_rnn_cell_175_matmul_1_readvariableop_resource*
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
while_body_11352697*
condR
while_cond_11352696*8
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
NoOpNoOp+^simple_rnn_cell_175/BiasAdd/ReadVariableOp*^simple_rnn_cell_175/MatMul/ReadVariableOp,^simple_rnn_cell_175/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2X
*simple_rnn_cell_175/BiasAdd/ReadVariableOp*simple_rnn_cell_175/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_175/MatMul/ReadVariableOp)simple_rnn_cell_175/MatMul/ReadVariableOp2Z
+simple_rnn_cell_175/MatMul_1/ReadVariableOp+simple_rnn_cell_175/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????P
"
_user_specified_name
inputs/0
?
f
H__inference_dropout_15_layer_call_and_return_conditional_losses_11352994

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
?
f
H__inference_dropout_15_layer_call_and_return_conditional_losses_11350983

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
?-
?
while_body_11351109
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_175_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_175_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_175_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_175/MatMul/ReadVariableOp?1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_175_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_175/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_175/BiasAddBiasAdd*while/simple_rnn_cell_175/MatMul:product:08while/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_175/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_175/addAddV2*while/simple_rnn_cell_175/BiasAdd:output:0,while/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_175/TanhTanh!while/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_175/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_175/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_175/MatMul/ReadVariableOp2^while/simple_rnn_cell_175/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_175_biasadd_readvariableop_resource;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_175_matmul_readvariableop_resource:while_simple_rnn_cell_175_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_175/MatMul/ReadVariableOp/while/simple_rnn_cell_175/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp: 
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
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11350318

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
?
?
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351450
simple_rnn_14_input(
simple_rnn_14_11351428:P$
simple_rnn_14_11351430:P(
simple_rnn_14_11351432:PP(
simple_rnn_15_11351436:Pd$
simple_rnn_15_11351438:d(
simple_rnn_15_11351440:dd"
dense_7_11351444:d
dense_7_11351446:
identity??dense_7/StatefulPartitionedCall?%simple_rnn_14/StatefulPartitionedCall?%simple_rnn_15/StatefulPartitionedCall?
%simple_rnn_14/StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_14_inputsimple_rnn_14_11351428simple_rnn_14_11351430simple_rnn_14_11351432*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11350848?
dropout_14/PartitionedCallPartitionedCall.simple_rnn_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_11350861?
%simple_rnn_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0simple_rnn_15_11351436simple_rnn_15_11351438simple_rnn_15_11351440*
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11350970?
dropout_15/PartitionedCallPartitionedCall.simple_rnn_15/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_11350983?
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_7_11351444dense_7_11351446*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_11350995w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_7/StatefulPartitionedCall&^simple_rnn_14/StatefulPartitionedCall&^simple_rnn_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2N
%simple_rnn_14/StatefulPartitionedCall%simple_rnn_14/StatefulPartitionedCall2N
%simple_rnn_15/StatefulPartitionedCall%simple_rnn_15/StatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_14_input
?4
?
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11350566

inputs.
simple_rnn_cell_175_11350491:Pd*
simple_rnn_cell_175_11350493:d.
simple_rnn_cell_175_11350495:dd
identity??+simple_rnn_cell_175/StatefulPartitionedCall?while;
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
+simple_rnn_cell_175/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_175_11350491simple_rnn_cell_175_11350493simple_rnn_cell_175_11350495*
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
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11350490n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_175_11350491simple_rnn_cell_175_11350493simple_rnn_cell_175_11350495*
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
while_body_11350503*
condR
while_cond_11350502*8
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
NoOpNoOp,^simple_rnn_cell_175/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????P: : : 2Z
+simple_rnn_cell_175/StatefulPartitionedCall+simple_rnn_cell_175/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????P
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11353149

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
?
?
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11353132

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
?4
?
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11350274

inputs.
simple_rnn_cell_174_11350199:P*
simple_rnn_cell_174_11350201:P.
simple_rnn_cell_174_11350203:PP
identity??+simple_rnn_cell_174/StatefulPartitionedCall?while;
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
+simple_rnn_cell_174/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_174_11350199simple_rnn_cell_174_11350201simple_rnn_cell_174_11350203*
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
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11350198n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_174_11350199simple_rnn_cell_174_11350201simple_rnn_cell_174_11350203*
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
while_body_11350211*
condR
while_cond_11350210*8
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
NoOpNoOp,^simple_rnn_cell_174/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_174/StatefulPartitionedCall+simple_rnn_cell_174/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
f
-__inference_dropout_14_layer_call_fn_11352486

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
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_11351204s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
while_cond_11352085
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11352085___redundant_placeholder06
2while_while_cond_11352085___redundant_placeholder16
2while_while_cond_11352085___redundant_placeholder26
2while_while_cond_11352085___redundant_placeholder3
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
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352152
inputs_0D
2simple_rnn_cell_174_matmul_readvariableop_resource:PA
3simple_rnn_cell_174_biasadd_readvariableop_resource:PF
4simple_rnn_cell_174_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_174/BiasAdd/ReadVariableOp?)simple_rnn_cell_174/MatMul/ReadVariableOp?+simple_rnn_cell_174/MatMul_1/ReadVariableOp?while=
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
)simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_174_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_174/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_174_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_174/BiasAddBiasAdd$simple_rnn_cell_174/MatMul:product:02simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_174_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_174/MatMul_1MatMulzeros:output:03simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_174/addAddV2$simple_rnn_cell_174/BiasAdd:output:0&simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_174/TanhTanhsimple_rnn_cell_174/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_174_matmul_readvariableop_resource3simple_rnn_cell_174_biasadd_readvariableop_resource4simple_rnn_cell_174_matmul_1_readvariableop_resource*
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
while_body_11352086*
condR
while_cond_11352085*8
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
NoOpNoOp+^simple_rnn_cell_174/BiasAdd/ReadVariableOp*^simple_rnn_cell_174/MatMul/ReadVariableOp,^simple_rnn_cell_174/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2X
*simple_rnn_cell_174/BiasAdd/ReadVariableOp*simple_rnn_cell_174/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_174/MatMul/ReadVariableOp)simple_rnn_cell_174/MatMul/ReadVariableOp2Z
+simple_rnn_cell_174/MatMul_1/ReadVariableOp+simple_rnn_cell_174/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
f
H__inference_dropout_14_layer_call_and_return_conditional_losses_11352491

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?=
?
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352979

inputsD
2simple_rnn_cell_175_matmul_readvariableop_resource:PdA
3simple_rnn_cell_175_biasadd_readvariableop_resource:dF
4simple_rnn_cell_175_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_175/BiasAdd/ReadVariableOp?)simple_rnn_cell_175/MatMul/ReadVariableOp?+simple_rnn_cell_175/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_175_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_175/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_175_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_175/BiasAddBiasAdd$simple_rnn_cell_175/MatMul:product:02simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_175_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_175/MatMul_1MatMulzeros:output:03simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_175/addAddV2$simple_rnn_cell_175/BiasAdd:output:0&simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_175/TanhTanhsimple_rnn_cell_175/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_175_matmul_readvariableop_resource3simple_rnn_cell_175_biasadd_readvariableop_resource4simple_rnn_cell_175_matmul_1_readvariableop_resource*
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
while_body_11352913*
condR
while_cond_11352912*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_175/BiasAdd/ReadVariableOp*^simple_rnn_cell_175/MatMul/ReadVariableOp,^simple_rnn_cell_175/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_175/BiasAdd/ReadVariableOp*simple_rnn_cell_175/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_175/MatMul/ReadVariableOp)simple_rnn_cell_175/MatMul/ReadVariableOp2Z
+simple_rnn_cell_175/MatMul_1/ReadVariableOp+simple_rnn_cell_175/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11350198

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
??
?	
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351743

inputsR
@simple_rnn_14_simple_rnn_cell_174_matmul_readvariableop_resource:PO
Asimple_rnn_14_simple_rnn_cell_174_biasadd_readvariableop_resource:PT
Bsimple_rnn_14_simple_rnn_cell_174_matmul_1_readvariableop_resource:PPR
@simple_rnn_15_simple_rnn_cell_175_matmul_readvariableop_resource:PdO
Asimple_rnn_15_simple_rnn_cell_175_biasadd_readvariableop_resource:dT
Bsimple_rnn_15_simple_rnn_cell_175_matmul_1_readvariableop_resource:dd8
&dense_7_matmul_readvariableop_resource:d5
'dense_7_biasadd_readvariableop_resource:
identity??dense_7/BiasAdd/ReadVariableOp?dense_7/MatMul/ReadVariableOp?8simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp?7simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp?9simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp?simple_rnn_14/while?8simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp?7simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp?9simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp?simple_rnn_15/whileI
simple_rnn_14/ShapeShapeinputs*
T0*
_output_shapes
:k
!simple_rnn_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_14/strided_sliceStridedSlicesimple_rnn_14/Shape:output:0*simple_rnn_14/strided_slice/stack:output:0,simple_rnn_14/strided_slice/stack_1:output:0,simple_rnn_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
simple_rnn_14/zeros/packedPack$simple_rnn_14/strided_slice:output:0%simple_rnn_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_14/zerosFill#simple_rnn_14/zeros/packed:output:0"simple_rnn_14/zeros/Const:output:0*
T0*'
_output_shapes
:?????????Pq
simple_rnn_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_14/transpose	Transposeinputs%simple_rnn_14/transpose/perm:output:0*
T0*+
_output_shapes
:?????????`
simple_rnn_14/Shape_1Shapesimple_rnn_14/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_14/strided_slice_1StridedSlicesimple_rnn_14/Shape_1:output:0,simple_rnn_14/strided_slice_1/stack:output:0.simple_rnn_14/strided_slice_1/stack_1:output:0.simple_rnn_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_14/TensorArrayV2TensorListReserve2simple_rnn_14/TensorArrayV2/element_shape:output:0&simple_rnn_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
5simple_rnn_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_14/transpose:y:0Lsimple_rnn_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_14/strided_slice_2StridedSlicesimple_rnn_14/transpose:y:0,simple_rnn_14/strided_slice_2/stack:output:0.simple_rnn_14/strided_slice_2/stack_1:output:0.simple_rnn_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
7simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp@simple_rnn_14_simple_rnn_cell_174_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
(simple_rnn_14/simple_rnn_cell_174/MatMulMatMul&simple_rnn_14/strided_slice_2:output:0?simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_14_simple_rnn_cell_174_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
)simple_rnn_14/simple_rnn_cell_174/BiasAddBiasAdd2simple_rnn_14/simple_rnn_cell_174/MatMul:product:0@simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
9simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_14_simple_rnn_cell_174_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
*simple_rnn_14/simple_rnn_cell_174/MatMul_1MatMulsimple_rnn_14/zeros:output:0Asimple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
%simple_rnn_14/simple_rnn_cell_174/addAddV22simple_rnn_14/simple_rnn_cell_174/BiasAdd:output:04simple_rnn_14/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
&simple_rnn_14/simple_rnn_cell_174/TanhTanh)simple_rnn_14/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P|
+simple_rnn_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
simple_rnn_14/TensorArrayV2_1TensorListReserve4simple_rnn_14/TensorArrayV2_1/element_shape:output:0&simple_rnn_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_14/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_14/whileWhile)simple_rnn_14/while/loop_counter:output:0/simple_rnn_14/while/maximum_iterations:output:0simple_rnn_14/time:output:0&simple_rnn_14/TensorArrayV2_1:handle:0simple_rnn_14/zeros:output:0&simple_rnn_14/strided_slice_1:output:0Esimple_rnn_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_14_simple_rnn_cell_174_matmul_readvariableop_resourceAsimple_rnn_14_simple_rnn_cell_174_biasadd_readvariableop_resourceBsimple_rnn_14_simple_rnn_cell_174_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *-
body%R#
!simple_rnn_14_while_body_11351565*-
cond%R#
!simple_rnn_14_while_cond_11351564*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
>simple_rnn_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
0simple_rnn_14/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_14/while:output:3Gsimple_rnn_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0v
#simple_rnn_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_14/strided_slice_3StridedSlice9simple_rnn_14/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_14/strided_slice_3/stack:output:0.simple_rnn_14/strided_slice_3/stack_1:output:0.simple_rnn_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_masks
simple_rnn_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_14/transpose_1	Transpose9simple_rnn_14/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_14/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????Pt
dropout_14/IdentityIdentitysimple_rnn_14/transpose_1:y:0*
T0*+
_output_shapes
:?????????P_
simple_rnn_15/ShapeShapedropout_14/Identity:output:0*
T0*
_output_shapes
:k
!simple_rnn_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#simple_rnn_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#simple_rnn_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_15/strided_sliceStridedSlicesimple_rnn_15/Shape:output:0*simple_rnn_15/strided_slice/stack:output:0,simple_rnn_15/strided_slice/stack_1:output:0,simple_rnn_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
simple_rnn_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
simple_rnn_15/zeros/packedPack$simple_rnn_15/strided_slice:output:0%simple_rnn_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:^
simple_rnn_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
simple_rnn_15/zerosFill#simple_rnn_15/zeros/packed:output:0"simple_rnn_15/zeros/Const:output:0*
T0*'
_output_shapes
:?????????dq
simple_rnn_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_15/transpose	Transposedropout_14/Identity:output:0%simple_rnn_15/transpose/perm:output:0*
T0*+
_output_shapes
:?????????P`
simple_rnn_15/Shape_1Shapesimple_rnn_15/transpose:y:0*
T0*
_output_shapes
:m
#simple_rnn_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_15/strided_slice_1StridedSlicesimple_rnn_15/Shape_1:output:0,simple_rnn_15/strided_slice_1/stack:output:0.simple_rnn_15/strided_slice_1/stack_1:output:0.simple_rnn_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
)simple_rnn_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
simple_rnn_15/TensorArrayV2TensorListReserve2simple_rnn_15/TensorArrayV2/element_shape:output:0&simple_rnn_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Csimple_rnn_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
5simple_rnn_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_15/transpose:y:0Lsimple_rnn_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???m
#simple_rnn_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%simple_rnn_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_15/strided_slice_2StridedSlicesimple_rnn_15/transpose:y:0,simple_rnn_15/strided_slice_2/stack:output:0.simple_rnn_15/strided_slice_2/stack_1:output:0.simple_rnn_15/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
7simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp@simple_rnn_15_simple_rnn_cell_175_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
(simple_rnn_15/simple_rnn_cell_175/MatMulMatMul&simple_rnn_15/strided_slice_2:output:0?simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOpAsimple_rnn_15_simple_rnn_cell_175_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
)simple_rnn_15/simple_rnn_cell_175/BiasAddBiasAdd2simple_rnn_15/simple_rnn_cell_175/MatMul:product:0@simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
9simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOpBsimple_rnn_15_simple_rnn_cell_175_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
*simple_rnn_15/simple_rnn_cell_175/MatMul_1MatMulsimple_rnn_15/zeros:output:0Asimple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
%simple_rnn_15/simple_rnn_cell_175/addAddV22simple_rnn_15/simple_rnn_cell_175/BiasAdd:output:04simple_rnn_15/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
&simple_rnn_15/simple_rnn_cell_175/TanhTanh)simple_rnn_15/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d|
+simple_rnn_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
simple_rnn_15/TensorArrayV2_1TensorListReserve4simple_rnn_15/TensorArrayV2_1/element_shape:output:0&simple_rnn_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???T
simple_rnn_15/timeConst*
_output_shapes
: *
dtype0*
value	B : q
&simple_rnn_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????b
 simple_rnn_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
simple_rnn_15/whileWhile)simple_rnn_15/while/loop_counter:output:0/simple_rnn_15/while/maximum_iterations:output:0simple_rnn_15/time:output:0&simple_rnn_15/TensorArrayV2_1:handle:0simple_rnn_15/zeros:output:0&simple_rnn_15/strided_slice_1:output:0Esimple_rnn_15/TensorArrayUnstack/TensorListFromTensor:output_handle:0@simple_rnn_15_simple_rnn_cell_175_matmul_readvariableop_resourceAsimple_rnn_15_simple_rnn_cell_175_biasadd_readvariableop_resourceBsimple_rnn_15_simple_rnn_cell_175_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *-
body%R#
!simple_rnn_15_while_body_11351670*-
cond%R#
!simple_rnn_15_while_cond_11351669*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
>simple_rnn_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
0simple_rnn_15/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_15/while:output:3Gsimple_rnn_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0v
#simple_rnn_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????o
%simple_rnn_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%simple_rnn_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
simple_rnn_15/strided_slice_3StridedSlice9simple_rnn_15/TensorArrayV2Stack/TensorListStack:tensor:0,simple_rnn_15/strided_slice_3/stack:output:0.simple_rnn_15/strided_slice_3/stack_1:output:0.simple_rnn_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_masks
simple_rnn_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
simple_rnn_15/transpose_1	Transpose9simple_rnn_15/TensorArrayV2Stack/TensorListStack:tensor:0'simple_rnn_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????dy
dropout_15/IdentityIdentity&simple_rnn_15/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
dense_7/MatMulMatMuldropout_15/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????g
IdentityIdentitydense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp9^simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp8^simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp:^simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp^simple_rnn_14/while9^simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp8^simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp:^simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp^simple_rnn_15/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2t
8simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp8simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp2r
7simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp7simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp2v
9simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp9simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp2*
simple_rnn_14/whilesimple_rnn_14/while2t
8simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp8simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp2r
7simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp7simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp2v
9simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp9simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp2*
simple_rnn_15/whilesimple_rnn_15/while:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_11352193
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11352193___redundant_placeholder06
2while_while_cond_11352193___redundant_placeholder16
2while_while_cond_11352193___redundant_placeholder26
2while_while_cond_11352193___redundant_placeholder3
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

?
!simple_rnn_15_while_cond_113518968
4simple_rnn_15_while_simple_rnn_15_while_loop_counter>
:simple_rnn_15_while_simple_rnn_15_while_maximum_iterations#
simple_rnn_15_while_placeholder%
!simple_rnn_15_while_placeholder_1%
!simple_rnn_15_while_placeholder_2:
6simple_rnn_15_while_less_simple_rnn_15_strided_slice_1R
Nsimple_rnn_15_while_simple_rnn_15_while_cond_11351896___redundant_placeholder0R
Nsimple_rnn_15_while_simple_rnn_15_while_cond_11351896___redundant_placeholder1R
Nsimple_rnn_15_while_simple_rnn_15_while_cond_11351896___redundant_placeholder2R
Nsimple_rnn_15_while_simple_rnn_15_while_cond_11351896___redundant_placeholder3 
simple_rnn_15_while_identity
?
simple_rnn_15/while/LessLesssimple_rnn_15_while_placeholder6simple_rnn_15_while_less_simple_rnn_15_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_15/while/IdentityIdentitysimple_rnn_15/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_15_while_identity%simple_rnn_15/while/Identity:output:0*(
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
/__inference_sequential_7_layer_call_fn_11351425
simple_rnn_14_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351385o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_14_input
?=
?
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11350848

inputsD
2simple_rnn_cell_174_matmul_readvariableop_resource:PA
3simple_rnn_cell_174_biasadd_readvariableop_resource:PF
4simple_rnn_cell_174_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_174/BiasAdd/ReadVariableOp?)simple_rnn_cell_174/MatMul/ReadVariableOp?+simple_rnn_cell_174/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_174_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_174/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_174_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_174/BiasAddBiasAdd$simple_rnn_cell_174/MatMul:product:02simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_174_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_174/MatMul_1MatMulzeros:output:03simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_174/addAddV2$simple_rnn_cell_174/BiasAdd:output:0&simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_174/TanhTanhsimple_rnn_cell_174/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_174_matmul_readvariableop_resource3simple_rnn_cell_174_biasadd_readvariableop_resource4simple_rnn_cell_174_matmul_1_readvariableop_resource*
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
while_body_11350782*
condR
while_cond_11350781*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_174/BiasAdd/ReadVariableOp*^simple_rnn_cell_174/MatMul/ReadVariableOp,^simple_rnn_cell_174/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_174/BiasAdd/ReadVariableOp*simple_rnn_cell_174/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_174/MatMul/ReadVariableOp)simple_rnn_cell_174/MatMul/ReadVariableOp2Z
+simple_rnn_cell_174/MatMul_1/ReadVariableOp+simple_rnn_cell_174/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_7_layer_call_fn_11351021
simple_rnn_14_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351002o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_14_input
?-
?
while_body_11352805
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_175_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_175_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_175_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_175/MatMul/ReadVariableOp?1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_175_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_175/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_175/BiasAddBiasAdd*while/simple_rnn_cell_175/MatMul:product:08while/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_175/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_175/addAddV2*while/simple_rnn_cell_175/BiasAdd:output:0,while/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_175/TanhTanh!while/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_175/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_175/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_175/MatMul/ReadVariableOp2^while/simple_rnn_cell_175/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_175_biasadd_readvariableop_resource;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_175_matmul_readvariableop_resource:while_simple_rnn_cell_175_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_175/MatMul/ReadVariableOp/while/simple_rnn_cell_175/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp: 
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
?:
?
!simple_rnn_14_while_body_113517858
4simple_rnn_14_while_simple_rnn_14_while_loop_counter>
:simple_rnn_14_while_simple_rnn_14_while_maximum_iterations#
simple_rnn_14_while_placeholder%
!simple_rnn_14_while_placeholder_1%
!simple_rnn_14_while_placeholder_27
3simple_rnn_14_while_simple_rnn_14_strided_slice_1_0s
osimple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_14_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource_0:PW
Isimple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource_0:P\
Jsimple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0:PP 
simple_rnn_14_while_identity"
simple_rnn_14_while_identity_1"
simple_rnn_14_while_identity_2"
simple_rnn_14_while_identity_3"
simple_rnn_14_while_identity_45
1simple_rnn_14_while_simple_rnn_14_strided_slice_1q
msimple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_14_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource:PU
Gsimple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource:PZ
Hsimple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource:PP??>simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp?=simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp??simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp?
Esimple_rnn_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
7simple_rnn_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_14_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_14_while_placeholderNsimple_rnn_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
=simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
.simple_rnn_14/while/simple_rnn_cell_174/MatMulMatMul>simple_rnn_14/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
/simple_rnn_14/while/simple_rnn_cell_174/BiasAddBiasAdd8simple_rnn_14/while/simple_rnn_cell_174/MatMul:product:0Fsimple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
?simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
0simple_rnn_14/while/simple_rnn_cell_174/MatMul_1MatMul!simple_rnn_14_while_placeholder_2Gsimple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_14/while/simple_rnn_cell_174/addAddV28simple_rnn_14/while/simple_rnn_cell_174/BiasAdd:output:0:simple_rnn_14/while/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
,simple_rnn_14/while/simple_rnn_cell_174/TanhTanh/simple_rnn_14/while/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_14_while_placeholder_1simple_rnn_14_while_placeholder0simple_rnn_14/while/simple_rnn_cell_174/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_14/while/addAddV2simple_rnn_14_while_placeholder"simple_rnn_14/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_14/while/add_1AddV24simple_rnn_14_while_simple_rnn_14_while_loop_counter$simple_rnn_14/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_14/while/IdentityIdentitysimple_rnn_14/while/add_1:z:0^simple_rnn_14/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_14/while/Identity_1Identity:simple_rnn_14_while_simple_rnn_14_while_maximum_iterations^simple_rnn_14/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_14/while/Identity_2Identitysimple_rnn_14/while/add:z:0^simple_rnn_14/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_14/while/Identity_3IdentityHsimple_rnn_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_14/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_14/while/Identity_4Identity0simple_rnn_14/while/simple_rnn_cell_174/Tanh:y:0^simple_rnn_14/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_14/while/NoOpNoOp?^simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp>^simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp@^simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_14_while_identity%simple_rnn_14/while/Identity:output:0"I
simple_rnn_14_while_identity_1'simple_rnn_14/while/Identity_1:output:0"I
simple_rnn_14_while_identity_2'simple_rnn_14/while/Identity_2:output:0"I
simple_rnn_14_while_identity_3'simple_rnn_14/while/Identity_3:output:0"I
simple_rnn_14_while_identity_4'simple_rnn_14/while/Identity_4:output:0"h
1simple_rnn_14_while_simple_rnn_14_strided_slice_13simple_rnn_14_while_simple_rnn_14_strided_slice_1_0"?
Gsimple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resourceIsimple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource_0"?
Hsimple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resourceJsimple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resourceHsimple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource_0"?
msimple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_14_tensorarrayunstack_tensorlistfromtensorosimple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
>simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp>simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp2~
=simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp=simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp2?
?simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp?simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp: 
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
while_cond_11350661
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11350661___redundant_placeholder06
2while_while_cond_11350661___redundant_placeholder16
2while_while_cond_11350661___redundant_placeholder26
2while_while_cond_11350661___redundant_placeholder3
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

?
!simple_rnn_15_while_cond_113516698
4simple_rnn_15_while_simple_rnn_15_while_loop_counter>
:simple_rnn_15_while_simple_rnn_15_while_maximum_iterations#
simple_rnn_15_while_placeholder%
!simple_rnn_15_while_placeholder_1%
!simple_rnn_15_while_placeholder_2:
6simple_rnn_15_while_less_simple_rnn_15_strided_slice_1R
Nsimple_rnn_15_while_simple_rnn_15_while_cond_11351669___redundant_placeholder0R
Nsimple_rnn_15_while_simple_rnn_15_while_cond_11351669___redundant_placeholder1R
Nsimple_rnn_15_while_simple_rnn_15_while_cond_11351669___redundant_placeholder2R
Nsimple_rnn_15_while_simple_rnn_15_while_cond_11351669___redundant_placeholder3 
simple_rnn_15_while_identity
?
simple_rnn_15/while/LessLesssimple_rnn_15_while_placeholder6simple_rnn_15_while_less_simple_rnn_15_strided_slice_1*
T0*
_output_shapes
: g
simple_rnn_15/while/IdentityIdentitysimple_rnn_15/while/Less:z:0*
T0
*
_output_shapes
: "E
simple_rnn_15_while_identity%simple_rnn_15/while/Identity:output:0*(
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
?=
?
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11351328

inputsD
2simple_rnn_cell_174_matmul_readvariableop_resource:PA
3simple_rnn_cell_174_biasadd_readvariableop_resource:PF
4simple_rnn_cell_174_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_174/BiasAdd/ReadVariableOp?)simple_rnn_cell_174/MatMul/ReadVariableOp?+simple_rnn_cell_174/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_174_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_174/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_174_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_174/BiasAddBiasAdd$simple_rnn_cell_174/MatMul:product:02simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_174_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_174/MatMul_1MatMulzeros:output:03simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_174/addAddV2$simple_rnn_cell_174/BiasAdd:output:0&simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_174/TanhTanhsimple_rnn_cell_174/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_174_matmul_readvariableop_resource3simple_rnn_cell_174_biasadd_readvariableop_resource4simple_rnn_cell_174_matmul_1_readvariableop_resource*
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
while_body_11351262*
condR
while_cond_11351261*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_174/BiasAdd/ReadVariableOp*^simple_rnn_cell_174/MatMul/ReadVariableOp,^simple_rnn_cell_174/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_174/BiasAdd/ReadVariableOp*simple_rnn_cell_174/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_174/MatMul/ReadVariableOp)simple_rnn_cell_174/MatMul/ReadVariableOp2Z
+simple_rnn_cell_174/MatMul_1/ReadVariableOp+simple_rnn_cell_174/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11350490

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
?
?
while_cond_11350369
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11350369___redundant_placeholder06
2while_while_cond_11350369___redundant_placeholder16
2while_while_cond_11350369___redundant_placeholder26
2while_while_cond_11350369___redundant_placeholder3
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
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11350433

inputs.
simple_rnn_cell_174_11350358:P*
simple_rnn_cell_174_11350360:P.
simple_rnn_cell_174_11350362:PP
identity??+simple_rnn_cell_174/StatefulPartitionedCall?while;
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
+simple_rnn_cell_174/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_174_11350358simple_rnn_cell_174_11350360simple_rnn_cell_174_11350362*
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
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11350318n
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_174_11350358simple_rnn_cell_174_11350360simple_rnn_cell_174_11350362*
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
while_body_11350370*
condR
while_cond_11350369*8
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
NoOpNoOp,^simple_rnn_cell_174/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????: : : 2Z
+simple_rnn_cell_174/StatefulPartitionedCall+simple_rnn_cell_174/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
0__inference_simple_rnn_14_layer_call_fn_11352033

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
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11350848s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
while_body_11352913
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_175_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_175_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_175_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_175/MatMul/ReadVariableOp?1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_175_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_175/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_175/BiasAddBiasAdd*while/simple_rnn_cell_175/MatMul:product:08while/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_175/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_175/addAddV2*while/simple_rnn_cell_175/BiasAdd:output:0,while/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_175/TanhTanh!while/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_175/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_175/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_175/MatMul/ReadVariableOp2^while/simple_rnn_cell_175/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_175_biasadd_readvariableop_resource;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_175_matmul_readvariableop_resource:while_simple_rnn_cell_175_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_175/MatMul/ReadVariableOp/while/simple_rnn_cell_175/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp: 
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
while_cond_11350502
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11350502___redundant_placeholder06
2while_while_cond_11350502___redundant_placeholder16
2while_while_cond_11350502___redundant_placeholder26
2while_while_cond_11350502___redundant_placeholder3
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
?
?
0__inference_simple_rnn_15_layer_call_fn_11352525
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11350725o
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
?=
?
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352871

inputsD
2simple_rnn_cell_175_matmul_readvariableop_resource:PdA
3simple_rnn_cell_175_biasadd_readvariableop_resource:dF
4simple_rnn_cell_175_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_175/BiasAdd/ReadVariableOp?)simple_rnn_cell_175/MatMul/ReadVariableOp?+simple_rnn_cell_175/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_175_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_175/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_175_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_175/BiasAddBiasAdd$simple_rnn_cell_175/MatMul:product:02simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_175_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_175/MatMul_1MatMulzeros:output:03simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_175/addAddV2$simple_rnn_cell_175/BiasAdd:output:0&simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_175/TanhTanhsimple_rnn_cell_175/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_175_matmul_readvariableop_resource3simple_rnn_cell_175_biasadd_readvariableop_resource4simple_rnn_cell_175_matmul_1_readvariableop_resource*
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
while_body_11352805*
condR
while_cond_11352804*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_175/BiasAdd/ReadVariableOp*^simple_rnn_cell_175/MatMul/ReadVariableOp,^simple_rnn_cell_175/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_175/BiasAdd/ReadVariableOp*simple_rnn_cell_175/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_175/MatMul/ReadVariableOp)simple_rnn_cell_175/MatMul/ReadVariableOp2Z
+simple_rnn_cell_175/MatMul_1/ReadVariableOp+simple_rnn_cell_175/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?

?
6__inference_simple_rnn_cell_174_layer_call_fn_11353039

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
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11350198o
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
?-
?
while_body_11350904
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_175_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_175_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_175_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_175/MatMul/ReadVariableOp?1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_175_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_175/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_175/BiasAddBiasAdd*while/simple_rnn_cell_175/MatMul:product:08while/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_175/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_175/addAddV2*while/simple_rnn_cell_175/BiasAdd:output:0,while/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_175/TanhTanh!while/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_175/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_175/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_175/MatMul/ReadVariableOp2^while/simple_rnn_cell_175/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_175_biasadd_readvariableop_resource;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_175_matmul_readvariableop_resource:while_simple_rnn_cell_175_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_175/MatMul/ReadVariableOp/while/simple_rnn_cell_175/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp: 
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
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11353087

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
?
?
while_cond_11350781
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11350781___redundant_placeholder06
2while_while_cond_11350781___redundant_placeholder16
2while_while_cond_11350781___redundant_placeholder26
2while_while_cond_11350781___redundant_placeholder3
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
?=
?
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11351175

inputsD
2simple_rnn_cell_175_matmul_readvariableop_resource:PdA
3simple_rnn_cell_175_biasadd_readvariableop_resource:dF
4simple_rnn_cell_175_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_175/BiasAdd/ReadVariableOp?)simple_rnn_cell_175/MatMul/ReadVariableOp?+simple_rnn_cell_175/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_175_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_175/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_175_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_175/BiasAddBiasAdd$simple_rnn_cell_175/MatMul:product:02simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_175_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_175/MatMul_1MatMulzeros:output:03simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_175/addAddV2$simple_rnn_cell_175/BiasAdd:output:0&simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_175/TanhTanhsimple_rnn_cell_175/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_175_matmul_readvariableop_resource3simple_rnn_cell_175_biasadd_readvariableop_resource4simple_rnn_cell_175_matmul_1_readvariableop_resource*
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
while_body_11351109*
condR
while_cond_11351108*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_175/BiasAdd/ReadVariableOp*^simple_rnn_cell_175/MatMul/ReadVariableOp,^simple_rnn_cell_175/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_175/BiasAdd/ReadVariableOp*simple_rnn_cell_175/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_175/MatMul/ReadVariableOp)simple_rnn_cell_175/MatMul/ReadVariableOp2Z
+simple_rnn_cell_175/MatMul_1/ReadVariableOp+simple_rnn_cell_175/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?=
?
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11350970

inputsD
2simple_rnn_cell_175_matmul_readvariableop_resource:PdA
3simple_rnn_cell_175_biasadd_readvariableop_resource:dF
4simple_rnn_cell_175_matmul_1_readvariableop_resource:dd
identity??*simple_rnn_cell_175/BiasAdd/ReadVariableOp?)simple_rnn_cell_175/MatMul/ReadVariableOp?+simple_rnn_cell_175/MatMul_1/ReadVariableOp?while;
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
:?????????PD
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
)simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_175_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
simple_rnn_cell_175/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
*simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_175_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
simple_rnn_cell_175/BiasAddBiasAdd$simple_rnn_cell_175/MatMul:product:02simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_175_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
simple_rnn_cell_175/MatMul_1MatMulzeros:output:03simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
simple_rnn_cell_175/addAddV2$simple_rnn_cell_175/BiasAdd:output:0&simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????do
simple_rnn_cell_175/TanhTanhsimple_rnn_cell_175/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_175_matmul_readvariableop_resource3simple_rnn_cell_175_biasadd_readvariableop_resource4simple_rnn_cell_175_matmul_1_readvariableop_resource*
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
while_body_11350904*
condR
while_cond_11350903*8
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
:?????????d*
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
:?????????dg
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:?????????d?
NoOpNoOp+^simple_rnn_cell_175/BiasAdd/ReadVariableOp*^simple_rnn_cell_175/MatMul/ReadVariableOp,^simple_rnn_cell_175/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????P: : : 2X
*simple_rnn_cell_175/BiasAdd/ReadVariableOp*simple_rnn_cell_175/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_175/MatMul/ReadVariableOp)simple_rnn_cell_175/MatMul/ReadVariableOp2Z
+simple_rnn_cell_175/MatMul_1/ReadVariableOp+simple_rnn_cell_175/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
.sequential_7_simple_rnn_15_while_cond_11350076R
Nsequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_while_loop_counterX
Tsequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_while_maximum_iterations0
,sequential_7_simple_rnn_15_while_placeholder2
.sequential_7_simple_rnn_15_while_placeholder_12
.sequential_7_simple_rnn_15_while_placeholder_2T
Psequential_7_simple_rnn_15_while_less_sequential_7_simple_rnn_15_strided_slice_1l
hsequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_while_cond_11350076___redundant_placeholder0l
hsequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_while_cond_11350076___redundant_placeholder1l
hsequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_while_cond_11350076___redundant_placeholder2l
hsequential_7_simple_rnn_15_while_sequential_7_simple_rnn_15_while_cond_11350076___redundant_placeholder3-
)sequential_7_simple_rnn_15_while_identity
?
%sequential_7/simple_rnn_15/while/LessLess,sequential_7_simple_rnn_15_while_placeholderPsequential_7_simple_rnn_15_while_less_sequential_7_simple_rnn_15_strided_slice_1*
T0*
_output_shapes
: ?
)sequential_7/simple_rnn_15/while/IdentityIdentity)sequential_7/simple_rnn_15/while/Less:z:0*
T0
*
_output_shapes
: "_
)sequential_7_simple_rnn_15_while_identity2sequential_7/simple_rnn_15/while/Identity:output:0*(
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
while_body_11352589
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_175_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_175_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_175_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_175/MatMul/ReadVariableOp?1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_175_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_175/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_175/BiasAddBiasAdd*while/simple_rnn_cell_175/MatMul:product:08while/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_175/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_175/addAddV2*while/simple_rnn_cell_175/BiasAdd:output:0,while/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_175/TanhTanh!while/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_175/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_175/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_175/MatMul/ReadVariableOp2^while/simple_rnn_cell_175/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_175_biasadd_readvariableop_resource;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_175_matmul_readvariableop_resource:while_simple_rnn_cell_175_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_175/MatMul/ReadVariableOp/while/simple_rnn_cell_175/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp: 
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
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352476

inputsD
2simple_rnn_cell_174_matmul_readvariableop_resource:PA
3simple_rnn_cell_174_biasadd_readvariableop_resource:PF
4simple_rnn_cell_174_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_174/BiasAdd/ReadVariableOp?)simple_rnn_cell_174/MatMul/ReadVariableOp?+simple_rnn_cell_174/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_174_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_174/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_174_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_174/BiasAddBiasAdd$simple_rnn_cell_174/MatMul:product:02simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_174_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_174/MatMul_1MatMulzeros:output:03simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_174/addAddV2$simple_rnn_cell_174/BiasAdd:output:0&simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_174/TanhTanhsimple_rnn_cell_174/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_174_matmul_readvariableop_resource3simple_rnn_cell_174_biasadd_readvariableop_resource4simple_rnn_cell_174_matmul_1_readvariableop_resource*
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
while_body_11352410*
condR
while_cond_11352409*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_174/BiasAdd/ReadVariableOp*^simple_rnn_cell_174/MatMul/ReadVariableOp,^simple_rnn_cell_174/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_174/BiasAdd/ReadVariableOp*simple_rnn_cell_174/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_174/MatMul/ReadVariableOp)simple_rnn_cell_174/MatMul/ReadVariableOp2Z
+simple_rnn_cell_174/MatMul_1/ReadVariableOp+simple_rnn_cell_174/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?:
?
!simple_rnn_15_while_body_113516708
4simple_rnn_15_while_simple_rnn_15_while_loop_counter>
:simple_rnn_15_while_simple_rnn_15_while_maximum_iterations#
simple_rnn_15_while_placeholder%
!simple_rnn_15_while_placeholder_1%
!simple_rnn_15_while_placeholder_27
3simple_rnn_15_while_simple_rnn_15_strided_slice_1_0s
osimple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_15_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource_0:PdW
Isimple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource_0:d\
Jsimple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0:dd 
simple_rnn_15_while_identity"
simple_rnn_15_while_identity_1"
simple_rnn_15_while_identity_2"
simple_rnn_15_while_identity_3"
simple_rnn_15_while_identity_45
1simple_rnn_15_while_simple_rnn_15_strided_slice_1q
msimple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_15_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource:PdU
Gsimple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource:dZ
Hsimple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource:dd??>simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp?=simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp??simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp?
Esimple_rnn_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
7simple_rnn_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_15_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_15_while_placeholderNsimple_rnn_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
=simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
.simple_rnn_15/while/simple_rnn_cell_175/MatMulMatMul>simple_rnn_15/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
/simple_rnn_15/while/simple_rnn_cell_175/BiasAddBiasAdd8simple_rnn_15/while/simple_rnn_cell_175/MatMul:product:0Fsimple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
?simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
0simple_rnn_15/while/simple_rnn_cell_175/MatMul_1MatMul!simple_rnn_15_while_placeholder_2Gsimple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_15/while/simple_rnn_cell_175/addAddV28simple_rnn_15/while/simple_rnn_cell_175/BiasAdd:output:0:simple_rnn_15/while/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
,simple_rnn_15/while/simple_rnn_cell_175/TanhTanh/simple_rnn_15/while/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_15_while_placeholder_1simple_rnn_15_while_placeholder0simple_rnn_15/while/simple_rnn_cell_175/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_15/while/addAddV2simple_rnn_15_while_placeholder"simple_rnn_15/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_15/while/add_1AddV24simple_rnn_15_while_simple_rnn_15_while_loop_counter$simple_rnn_15/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_15/while/IdentityIdentitysimple_rnn_15/while/add_1:z:0^simple_rnn_15/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_15/while/Identity_1Identity:simple_rnn_15_while_simple_rnn_15_while_maximum_iterations^simple_rnn_15/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_15/while/Identity_2Identitysimple_rnn_15/while/add:z:0^simple_rnn_15/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_15/while/Identity_3IdentityHsimple_rnn_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_15/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_15/while/Identity_4Identity0simple_rnn_15/while/simple_rnn_cell_175/Tanh:y:0^simple_rnn_15/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_15/while/NoOpNoOp?^simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp>^simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp@^simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_15_while_identity%simple_rnn_15/while/Identity:output:0"I
simple_rnn_15_while_identity_1'simple_rnn_15/while/Identity_1:output:0"I
simple_rnn_15_while_identity_2'simple_rnn_15/while/Identity_2:output:0"I
simple_rnn_15_while_identity_3'simple_rnn_15/while/Identity_3:output:0"I
simple_rnn_15_while_identity_4'simple_rnn_15/while/Identity_4:output:0"h
1simple_rnn_15_while_simple_rnn_15_strided_slice_13simple_rnn_15_while_simple_rnn_15_strided_slice_1_0"?
Gsimple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resourceIsimple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource_0"?
Hsimple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resourceJsimple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resourceHsimple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource_0"?
msimple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_15_tensorarrayunstack_tensorlistfromtensorosimple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
>simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp>simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp2~
=simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp=simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp2?
?simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp?simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp: 
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
?
?
0__inference_simple_rnn_15_layer_call_fn_11352536

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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11350970o
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
:?????????P: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?=
?
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352368

inputsD
2simple_rnn_cell_174_matmul_readvariableop_resource:PA
3simple_rnn_cell_174_biasadd_readvariableop_resource:PF
4simple_rnn_cell_174_matmul_1_readvariableop_resource:PP
identity??*simple_rnn_cell_174/BiasAdd/ReadVariableOp?)simple_rnn_cell_174/MatMul/ReadVariableOp?+simple_rnn_cell_174/MatMul_1/ReadVariableOp?while;
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
:?????????D
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
)simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp2simple_rnn_cell_174_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
simple_rnn_cell_174/MatMulMatMulstrided_slice_2:output:01simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
*simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp3simple_rnn_cell_174_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
simple_rnn_cell_174/BiasAddBiasAdd$simple_rnn_cell_174/MatMul:product:02simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp4simple_rnn_cell_174_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
simple_rnn_cell_174/MatMul_1MatMulzeros:output:03simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
simple_rnn_cell_174/addAddV2$simple_rnn_cell_174/BiasAdd:output:0&simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????Po
simple_rnn_cell_174/TanhTanhsimple_rnn_cell_174/add:z:0*
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:02simple_rnn_cell_174_matmul_readvariableop_resource3simple_rnn_cell_174_biasadd_readvariableop_resource4simple_rnn_cell_174_matmul_1_readvariableop_resource*
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
while_body_11352302*
condR
while_cond_11352301*8
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
:?????????P*
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
:?????????Pb
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:?????????P?
NoOpNoOp+^simple_rnn_cell_174/BiasAdd/ReadVariableOp*^simple_rnn_cell_174/MatMul/ReadVariableOp,^simple_rnn_cell_174/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 2X
*simple_rnn_cell_174/BiasAdd/ReadVariableOp*simple_rnn_cell_174/BiasAdd/ReadVariableOp2V
)simple_rnn_cell_174/MatMul/ReadVariableOp)simple_rnn_cell_174/MatMul/ReadVariableOp2Z
+simple_rnn_cell_174/MatMul_1/ReadVariableOp+simple_rnn_cell_174/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_11352696
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11352696___redundant_placeholder06
2while_while_cond_11352696___redundant_placeholder16
2while_while_cond_11352696___redundant_placeholder26
2while_while_cond_11352696___redundant_placeholder3
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
while_cond_11352912
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11352912___redundant_placeholder06
2while_while_cond_11352912___redundant_placeholder16
2while_while_cond_11352912___redundant_placeholder26
2while_while_cond_11352912___redundant_placeholder3
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
?F
?
.sequential_7_simple_rnn_14_while_body_11349972R
Nsequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_while_loop_counterX
Tsequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_while_maximum_iterations0
,sequential_7_simple_rnn_14_while_placeholder2
.sequential_7_simple_rnn_14_while_placeholder_12
.sequential_7_simple_rnn_14_while_placeholder_2Q
Msequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_strided_slice_1_0?
?sequential_7_simple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_14_tensorarrayunstack_tensorlistfromtensor_0g
Usequential_7_simple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource_0:Pd
Vsequential_7_simple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource_0:Pi
Wsequential_7_simple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0:PP-
)sequential_7_simple_rnn_14_while_identity/
+sequential_7_simple_rnn_14_while_identity_1/
+sequential_7_simple_rnn_14_while_identity_2/
+sequential_7_simple_rnn_14_while_identity_3/
+sequential_7_simple_rnn_14_while_identity_4O
Ksequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_strided_slice_1?
?sequential_7_simple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_14_tensorarrayunstack_tensorlistfromtensore
Ssequential_7_simple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource:Pb
Tsequential_7_simple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource:Pg
Usequential_7_simple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource:PP??Ksequential_7/simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp?Jsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp?Lsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp?
Rsequential_7/simple_rnn_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Dsequential_7/simple_rnn_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem?sequential_7_simple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_14_tensorarrayunstack_tensorlistfromtensor_0,sequential_7_simple_rnn_14_while_placeholder[sequential_7/simple_rnn_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
Jsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOpUsequential_7_simple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
;sequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMulMatMulKsequential_7/simple_rnn_14/while/TensorArrayV2Read/TensorListGetItem:item:0Rsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Ksequential_7/simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOpVsequential_7_simple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
<sequential_7/simple_rnn_14/while/simple_rnn_cell_174/BiasAddBiasAddEsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul:product:0Ssequential_7/simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Lsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOpWsequential_7_simple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
=sequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul_1MatMul.sequential_7_simple_rnn_14_while_placeholder_2Tsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
8sequential_7/simple_rnn_14/while/simple_rnn_cell_174/addAddV2Esequential_7/simple_rnn_14/while/simple_rnn_cell_174/BiasAdd:output:0Gsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
9sequential_7/simple_rnn_14/while/simple_rnn_cell_174/TanhTanh<sequential_7/simple_rnn_14/while/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P?
Esequential_7/simple_rnn_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem.sequential_7_simple_rnn_14_while_placeholder_1,sequential_7_simple_rnn_14_while_placeholder=sequential_7/simple_rnn_14/while/simple_rnn_cell_174/Tanh:y:0*
_output_shapes
: *
element_dtype0:???h
&sequential_7/simple_rnn_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
$sequential_7/simple_rnn_14/while/addAddV2,sequential_7_simple_rnn_14_while_placeholder/sequential_7/simple_rnn_14/while/add/y:output:0*
T0*
_output_shapes
: j
(sequential_7/simple_rnn_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
&sequential_7/simple_rnn_14/while/add_1AddV2Nsequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_while_loop_counter1sequential_7/simple_rnn_14/while/add_1/y:output:0*
T0*
_output_shapes
: ?
)sequential_7/simple_rnn_14/while/IdentityIdentity*sequential_7/simple_rnn_14/while/add_1:z:0&^sequential_7/simple_rnn_14/while/NoOp*
T0*
_output_shapes
: ?
+sequential_7/simple_rnn_14/while/Identity_1IdentityTsequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_while_maximum_iterations&^sequential_7/simple_rnn_14/while/NoOp*
T0*
_output_shapes
: ?
+sequential_7/simple_rnn_14/while/Identity_2Identity(sequential_7/simple_rnn_14/while/add:z:0&^sequential_7/simple_rnn_14/while/NoOp*
T0*
_output_shapes
: ?
+sequential_7/simple_rnn_14/while/Identity_3IdentityUsequential_7/simple_rnn_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0&^sequential_7/simple_rnn_14/while/NoOp*
T0*
_output_shapes
: :????
+sequential_7/simple_rnn_14/while/Identity_4Identity=sequential_7/simple_rnn_14/while/simple_rnn_cell_174/Tanh:y:0&^sequential_7/simple_rnn_14/while/NoOp*
T0*'
_output_shapes
:?????????P?
%sequential_7/simple_rnn_14/while/NoOpNoOpL^sequential_7/simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOpK^sequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOpM^sequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "_
)sequential_7_simple_rnn_14_while_identity2sequential_7/simple_rnn_14/while/Identity:output:0"c
+sequential_7_simple_rnn_14_while_identity_14sequential_7/simple_rnn_14/while/Identity_1:output:0"c
+sequential_7_simple_rnn_14_while_identity_24sequential_7/simple_rnn_14/while/Identity_2:output:0"c
+sequential_7_simple_rnn_14_while_identity_34sequential_7/simple_rnn_14/while/Identity_3:output:0"c
+sequential_7_simple_rnn_14_while_identity_44sequential_7/simple_rnn_14/while/Identity_4:output:0"?
Ksequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_strided_slice_1Msequential_7_simple_rnn_14_while_sequential_7_simple_rnn_14_strided_slice_1_0"?
Tsequential_7_simple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resourceVsequential_7_simple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource_0"?
Usequential_7_simple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resourceWsequential_7_simple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0"?
Ssequential_7_simple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resourceUsequential_7_simple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource_0"?
?sequential_7_simple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_14_tensorarrayunstack_tensorlistfromtensor?sequential_7_simple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_simple_rnn_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
Ksequential_7/simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOpKsequential_7/simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp2?
Jsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOpJsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp2?
Lsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOpLsequential_7/simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp: 
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
?
&__inference_signature_wrapper_11352000
simple_rnn_14_input
unknown:P
	unknown_0:P
	unknown_1:PP
	unknown_2:Pd
	unknown_3:d
	unknown_4:dd
	unknown_5:d
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallsimple_rnn_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_11350150o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_14_input
?

?
6__inference_simple_rnn_cell_175_layer_call_fn_11353115

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
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11350610o
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
?
?
0__inference_simple_rnn_14_layer_call_fn_11352044

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
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11351328s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_11350210
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11350210___redundant_placeholder06
2while_while_cond_11350210___redundant_placeholder16
2while_while_cond_11350210___redundant_placeholder26
2while_while_cond_11350210___redundant_placeholder3
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
?
f
H__inference_dropout_14_layer_call_and_return_conditional_losses_11350861

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????P_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????P"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
while_cond_11351261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11351261___redundant_placeholder06
2while_while_cond_11351261___redundant_placeholder16
2while_while_cond_11351261___redundant_placeholder26
2while_while_cond_11351261___redundant_placeholder3
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

?
6__inference_simple_rnn_cell_174_layer_call_fn_11353053

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
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11350318o
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
?-
?
while_body_11350782
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_174_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_174_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_174_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_174/MatMul/ReadVariableOp?1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_174_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_174/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_174/BiasAddBiasAdd*while/simple_rnn_cell_174/MatMul:product:08while/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_174/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_174/addAddV2*while/simple_rnn_cell_174/BiasAdd:output:0,while/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_174/TanhTanh!while/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_174/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_174/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_174/MatMul/ReadVariableOp2^while/simple_rnn_cell_174/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_174_biasadd_readvariableop_resource;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_174_matmul_readvariableop_resource:while_simple_rnn_cell_174_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_174/MatMul/ReadVariableOp/while/simple_rnn_cell_174/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp: 
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
?!
?
while_body_11350662
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_175_11350684_0:Pd2
$while_simple_rnn_cell_175_11350686_0:d6
$while_simple_rnn_cell_175_11350688_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_175_11350684:Pd0
"while_simple_rnn_cell_175_11350686:d4
"while_simple_rnn_cell_175_11350688:dd??1while/simple_rnn_cell_175/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
1while/simple_rnn_cell_175/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_175_11350684_0$while_simple_rnn_cell_175_11350686_0$while_simple_rnn_cell_175_11350688_0*
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
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11350610?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_175/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_175/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp2^while/simple_rnn_cell_175/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_175_11350684$while_simple_rnn_cell_175_11350684_0"J
"while_simple_rnn_cell_175_11350686$while_simple_rnn_cell_175_11350686_0"J
"while_simple_rnn_cell_175_11350688$while_simple_rnn_cell_175_11350688_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2f
1while/simple_rnn_cell_175/StatefulPartitionedCall1while/simple_rnn_cell_175/StatefulPartitionedCall: 
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
?
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351385

inputs(
simple_rnn_14_11351363:P$
simple_rnn_14_11351365:P(
simple_rnn_14_11351367:PP(
simple_rnn_15_11351371:Pd$
simple_rnn_15_11351373:d(
simple_rnn_15_11351375:dd"
dense_7_11351379:d
dense_7_11351381:
identity??dense_7/StatefulPartitionedCall?"dropout_14/StatefulPartitionedCall?"dropout_15/StatefulPartitionedCall?%simple_rnn_14/StatefulPartitionedCall?%simple_rnn_15/StatefulPartitionedCall?
%simple_rnn_14/StatefulPartitionedCallStatefulPartitionedCallinputssimple_rnn_14_11351363simple_rnn_14_11351365simple_rnn_14_11351367*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11351328?
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????P* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_11351204?
%simple_rnn_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0simple_rnn_15_11351371simple_rnn_15_11351373simple_rnn_15_11351375*
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11351175?
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall.simple_rnn_15/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_11351051?
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_7_11351379dense_7_11351381*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_11350995w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp ^dense_7/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall&^simple_rnn_14/StatefulPartitionedCall&^simple_rnn_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2N
%simple_rnn_14/StatefulPartitionedCall%simple_rnn_14/StatefulPartitionedCall2N
%simple_rnn_15/StatefulPartitionedCall%simple_rnn_15/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_7_layer_call_and_return_conditional_losses_11353025

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
?	
?
/__inference_sequential_7_layer_call_fn_11351523

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
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351385o
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
':?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

g
H__inference_dropout_14_layer_call_and_return_conditional_losses_11352503

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
:?????????PC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????P*
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
:?????????Ps
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????Pm
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????P]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????P"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????P:S O
+
_output_shapes
:?????????P
 
_user_specified_nameinputs
?
?
while_cond_11352588
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11352588___redundant_placeholder06
2while_while_cond_11352588___redundant_placeholder16
2while_while_cond_11352588___redundant_placeholder26
2while_while_cond_11352588___redundant_placeholder3
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
while_body_11352697
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_175_matmul_readvariableop_resource_0:PdI
;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0:dN
<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0:dd
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_175_matmul_readvariableop_resource:PdG
9while_simple_rnn_cell_175_biasadd_readvariableop_resource:dL
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource:dd??0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_175/MatMul/ReadVariableOp?1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
/while/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_175_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
 while/simple_rnn_cell_175/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
!while/simple_rnn_cell_175/BiasAddBiasAdd*while/simple_rnn_cell_175/MatMul:product:08while/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
"while/simple_rnn_cell_175/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
while/simple_rnn_cell_175/addAddV2*while/simple_rnn_cell_175/BiasAdd:output:0,while/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d{
while/simple_rnn_cell_175/TanhTanh!while/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_175/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_175/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????d?

while/NoOpNoOp1^while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_175/MatMul/ReadVariableOp2^while/simple_rnn_cell_175/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_175_biasadd_readvariableop_resource;while_simple_rnn_cell_175_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_175_matmul_1_readvariableop_resource<while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_175_matmul_readvariableop_resource:while_simple_rnn_cell_175_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2d
0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp0while/simple_rnn_cell_175/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_175/MatMul/ReadVariableOp/while/simple_rnn_cell_175/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp1while/simple_rnn_cell_175/MatMul_1/ReadVariableOp: 
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
?
?
0__inference_simple_rnn_15_layer_call_fn_11352514
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11350566o
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
?
?
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11353070

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
?:
?
!simple_rnn_14_while_body_113515658
4simple_rnn_14_while_simple_rnn_14_while_loop_counter>
:simple_rnn_14_while_simple_rnn_14_while_maximum_iterations#
simple_rnn_14_while_placeholder%
!simple_rnn_14_while_placeholder_1%
!simple_rnn_14_while_placeholder_27
3simple_rnn_14_while_simple_rnn_14_strided_slice_1_0s
osimple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_14_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource_0:PW
Isimple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource_0:P\
Jsimple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0:PP 
simple_rnn_14_while_identity"
simple_rnn_14_while_identity_1"
simple_rnn_14_while_identity_2"
simple_rnn_14_while_identity_3"
simple_rnn_14_while_identity_45
1simple_rnn_14_while_simple_rnn_14_strided_slice_1q
msimple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_14_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource:PU
Gsimple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource:PZ
Hsimple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource:PP??>simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp?=simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp??simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp?
Esimple_rnn_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
7simple_rnn_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_14_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_14_while_placeholderNsimple_rnn_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
=simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
.simple_rnn_14/while/simple_rnn_cell_174/MatMulMatMul>simple_rnn_14/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
>simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
/simple_rnn_14/while/simple_rnn_cell_174/BiasAddBiasAdd8simple_rnn_14/while/simple_rnn_cell_174/MatMul:product:0Fsimple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
?simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
0simple_rnn_14/while/simple_rnn_cell_174/MatMul_1MatMul!simple_rnn_14_while_placeholder_2Gsimple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
+simple_rnn_14/while/simple_rnn_cell_174/addAddV28simple_rnn_14/while/simple_rnn_cell_174/BiasAdd:output:0:simple_rnn_14/while/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
,simple_rnn_14/while/simple_rnn_cell_174/TanhTanh/simple_rnn_14/while/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P?
8simple_rnn_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_14_while_placeholder_1simple_rnn_14_while_placeholder0simple_rnn_14/while/simple_rnn_cell_174/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_14/while/addAddV2simple_rnn_14_while_placeholder"simple_rnn_14/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_14/while/add_1AddV24simple_rnn_14_while_simple_rnn_14_while_loop_counter$simple_rnn_14/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_14/while/IdentityIdentitysimple_rnn_14/while/add_1:z:0^simple_rnn_14/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_14/while/Identity_1Identity:simple_rnn_14_while_simple_rnn_14_while_maximum_iterations^simple_rnn_14/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_14/while/Identity_2Identitysimple_rnn_14/while/add:z:0^simple_rnn_14/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_14/while/Identity_3IdentityHsimple_rnn_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_14/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_14/while/Identity_4Identity0simple_rnn_14/while/simple_rnn_cell_174/Tanh:y:0^simple_rnn_14/while/NoOp*
T0*'
_output_shapes
:?????????P?
simple_rnn_14/while/NoOpNoOp?^simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp>^simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp@^simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_14_while_identity%simple_rnn_14/while/Identity:output:0"I
simple_rnn_14_while_identity_1'simple_rnn_14/while/Identity_1:output:0"I
simple_rnn_14_while_identity_2'simple_rnn_14/while/Identity_2:output:0"I
simple_rnn_14_while_identity_3'simple_rnn_14/while/Identity_3:output:0"I
simple_rnn_14_while_identity_4'simple_rnn_14/while/Identity_4:output:0"h
1simple_rnn_14_while_simple_rnn_14_strided_slice_13simple_rnn_14_while_simple_rnn_14_strided_slice_1_0"?
Gsimple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resourceIsimple_rnn_14_while_simple_rnn_cell_174_biasadd_readvariableop_resource_0"?
Hsimple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resourceJsimple_rnn_14_while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resourceHsimple_rnn_14_while_simple_rnn_cell_174_matmul_readvariableop_resource_0"?
msimple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_14_tensorarrayunstack_tensorlistfromtensorosimple_rnn_14_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2?
>simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp>simple_rnn_14/while/simple_rnn_cell_174/BiasAdd/ReadVariableOp2~
=simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp=simple_rnn_14/while/simple_rnn_cell_174/MatMul/ReadVariableOp2?
?simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp?simple_rnn_14/while/simple_rnn_cell_174/MatMul_1/ReadVariableOp: 
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
0__inference_simple_rnn_14_layer_call_fn_11352011
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
GPU 2J 8? *T
fORM
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11350274|
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
ѷ
?

#__inference__wrapped_model_11350150
simple_rnn_14_input_
Msequential_7_simple_rnn_14_simple_rnn_cell_174_matmul_readvariableop_resource:P\
Nsequential_7_simple_rnn_14_simple_rnn_cell_174_biasadd_readvariableop_resource:Pa
Osequential_7_simple_rnn_14_simple_rnn_cell_174_matmul_1_readvariableop_resource:PP_
Msequential_7_simple_rnn_15_simple_rnn_cell_175_matmul_readvariableop_resource:Pd\
Nsequential_7_simple_rnn_15_simple_rnn_cell_175_biasadd_readvariableop_resource:da
Osequential_7_simple_rnn_15_simple_rnn_cell_175_matmul_1_readvariableop_resource:ddE
3sequential_7_dense_7_matmul_readvariableop_resource:dB
4sequential_7_dense_7_biasadd_readvariableop_resource:
identity??+sequential_7/dense_7/BiasAdd/ReadVariableOp?*sequential_7/dense_7/MatMul/ReadVariableOp?Esequential_7/simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp?Dsequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp?Fsequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp? sequential_7/simple_rnn_14/while?Esequential_7/simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp?Dsequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp?Fsequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp? sequential_7/simple_rnn_15/whilec
 sequential_7/simple_rnn_14/ShapeShapesimple_rnn_14_input*
T0*
_output_shapes
:x
.sequential_7/simple_rnn_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_7/simple_rnn_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_7/simple_rnn_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_7/simple_rnn_14/strided_sliceStridedSlice)sequential_7/simple_rnn_14/Shape:output:07sequential_7/simple_rnn_14/strided_slice/stack:output:09sequential_7/simple_rnn_14/strided_slice/stack_1:output:09sequential_7/simple_rnn_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_7/simple_rnn_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :P?
'sequential_7/simple_rnn_14/zeros/packedPack1sequential_7/simple_rnn_14/strided_slice:output:02sequential_7/simple_rnn_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_7/simple_rnn_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 sequential_7/simple_rnn_14/zerosFill0sequential_7/simple_rnn_14/zeros/packed:output:0/sequential_7/simple_rnn_14/zeros/Const:output:0*
T0*'
_output_shapes
:?????????P~
)sequential_7/simple_rnn_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
$sequential_7/simple_rnn_14/transpose	Transposesimple_rnn_14_input2sequential_7/simple_rnn_14/transpose/perm:output:0*
T0*+
_output_shapes
:?????????z
"sequential_7/simple_rnn_14/Shape_1Shape(sequential_7/simple_rnn_14/transpose:y:0*
T0*
_output_shapes
:z
0sequential_7/simple_rnn_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_7/simple_rnn_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_7/simple_rnn_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_7/simple_rnn_14/strided_slice_1StridedSlice+sequential_7/simple_rnn_14/Shape_1:output:09sequential_7/simple_rnn_14/strided_slice_1/stack:output:0;sequential_7/simple_rnn_14/strided_slice_1/stack_1:output:0;sequential_7/simple_rnn_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6sequential_7/simple_rnn_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
(sequential_7/simple_rnn_14/TensorArrayV2TensorListReserve?sequential_7/simple_rnn_14/TensorArrayV2/element_shape:output:03sequential_7/simple_rnn_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Psequential_7/simple_rnn_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
Bsequential_7/simple_rnn_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_7/simple_rnn_14/transpose:y:0Ysequential_7/simple_rnn_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???z
0sequential_7/simple_rnn_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_7/simple_rnn_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_7/simple_rnn_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_7/simple_rnn_14/strided_slice_2StridedSlice(sequential_7/simple_rnn_14/transpose:y:09sequential_7/simple_rnn_14/strided_slice_2/stack:output:0;sequential_7/simple_rnn_14/strided_slice_2/stack_1:output:0;sequential_7/simple_rnn_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask?
Dsequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOpMsequential_7_simple_rnn_14_simple_rnn_cell_174_matmul_readvariableop_resource*
_output_shapes

:P*
dtype0?
5sequential_7/simple_rnn_14/simple_rnn_cell_174/MatMulMatMul3sequential_7/simple_rnn_14/strided_slice_2:output:0Lsequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Esequential_7/simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOpNsequential_7_simple_rnn_14_simple_rnn_cell_174_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0?
6sequential_7/simple_rnn_14/simple_rnn_cell_174/BiasAddBiasAdd?sequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul:product:0Msequential_7/simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
Fsequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOpOsequential_7_simple_rnn_14_simple_rnn_cell_174_matmul_1_readvariableop_resource*
_output_shapes

:PP*
dtype0?
7sequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul_1MatMul)sequential_7/simple_rnn_14/zeros:output:0Nsequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
2sequential_7/simple_rnn_14/simple_rnn_cell_174/addAddV2?sequential_7/simple_rnn_14/simple_rnn_cell_174/BiasAdd:output:0Asequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P?
3sequential_7/simple_rnn_14/simple_rnn_cell_174/TanhTanh6sequential_7/simple_rnn_14/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P?
8sequential_7/simple_rnn_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
*sequential_7/simple_rnn_14/TensorArrayV2_1TensorListReserveAsequential_7/simple_rnn_14/TensorArrayV2_1/element_shape:output:03sequential_7/simple_rnn_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???a
sequential_7/simple_rnn_14/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_7/simple_rnn_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
-sequential_7/simple_rnn_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
 sequential_7/simple_rnn_14/whileWhile6sequential_7/simple_rnn_14/while/loop_counter:output:0<sequential_7/simple_rnn_14/while/maximum_iterations:output:0(sequential_7/simple_rnn_14/time:output:03sequential_7/simple_rnn_14/TensorArrayV2_1:handle:0)sequential_7/simple_rnn_14/zeros:output:03sequential_7/simple_rnn_14/strided_slice_1:output:0Rsequential_7/simple_rnn_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_7_simple_rnn_14_simple_rnn_cell_174_matmul_readvariableop_resourceNsequential_7_simple_rnn_14_simple_rnn_cell_174_biasadd_readvariableop_resourceOsequential_7_simple_rnn_14_simple_rnn_cell_174_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *:
body2R0
.sequential_7_simple_rnn_14_while_body_11349972*:
cond2R0
.sequential_7_simple_rnn_14_while_cond_11349971*8
output_shapes'
%: : : : :?????????P: : : : : *
parallel_iterations ?
Ksequential_7/simple_rnn_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
=sequential_7/simple_rnn_14/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_7/simple_rnn_14/while:output:3Tsequential_7/simple_rnn_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????P*
element_dtype0?
0sequential_7/simple_rnn_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????|
2sequential_7/simple_rnn_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_7/simple_rnn_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_7/simple_rnn_14/strided_slice_3StridedSliceFsequential_7/simple_rnn_14/TensorArrayV2Stack/TensorListStack:tensor:09sequential_7/simple_rnn_14/strided_slice_3/stack:output:0;sequential_7/simple_rnn_14/strided_slice_3/stack_1:output:0;sequential_7/simple_rnn_14/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
+sequential_7/simple_rnn_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
&sequential_7/simple_rnn_14/transpose_1	TransposeFsequential_7/simple_rnn_14/TensorArrayV2Stack/TensorListStack:tensor:04sequential_7/simple_rnn_14/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????P?
 sequential_7/dropout_14/IdentityIdentity*sequential_7/simple_rnn_14/transpose_1:y:0*
T0*+
_output_shapes
:?????????Py
 sequential_7/simple_rnn_15/ShapeShape)sequential_7/dropout_14/Identity:output:0*
T0*
_output_shapes
:x
.sequential_7/simple_rnn_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0sequential_7/simple_rnn_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0sequential_7/simple_rnn_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential_7/simple_rnn_15/strided_sliceStridedSlice)sequential_7/simple_rnn_15/Shape:output:07sequential_7/simple_rnn_15/strided_slice/stack:output:09sequential_7/simple_rnn_15/strided_slice/stack_1:output:09sequential_7/simple_rnn_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
)sequential_7/simple_rnn_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d?
'sequential_7/simple_rnn_15/zeros/packedPack1sequential_7/simple_rnn_15/strided_slice:output:02sequential_7/simple_rnn_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&sequential_7/simple_rnn_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
 sequential_7/simple_rnn_15/zerosFill0sequential_7/simple_rnn_15/zeros/packed:output:0/sequential_7/simple_rnn_15/zeros/Const:output:0*
T0*'
_output_shapes
:?????????d~
)sequential_7/simple_rnn_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
$sequential_7/simple_rnn_15/transpose	Transpose)sequential_7/dropout_14/Identity:output:02sequential_7/simple_rnn_15/transpose/perm:output:0*
T0*+
_output_shapes
:?????????Pz
"sequential_7/simple_rnn_15/Shape_1Shape(sequential_7/simple_rnn_15/transpose:y:0*
T0*
_output_shapes
:z
0sequential_7/simple_rnn_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_7/simple_rnn_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_7/simple_rnn_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_7/simple_rnn_15/strided_slice_1StridedSlice+sequential_7/simple_rnn_15/Shape_1:output:09sequential_7/simple_rnn_15/strided_slice_1/stack:output:0;sequential_7/simple_rnn_15/strided_slice_1/stack_1:output:0;sequential_7/simple_rnn_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
6sequential_7/simple_rnn_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
??????????
(sequential_7/simple_rnn_15/TensorArrayV2TensorListReserve?sequential_7/simple_rnn_15/TensorArrayV2/element_shape:output:03sequential_7/simple_rnn_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:????
Psequential_7/simple_rnn_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
Bsequential_7/simple_rnn_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor(sequential_7/simple_rnn_15/transpose:y:0Ysequential_7/simple_rnn_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???z
0sequential_7/simple_rnn_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2sequential_7/simple_rnn_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2sequential_7/simple_rnn_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_7/simple_rnn_15/strided_slice_2StridedSlice(sequential_7/simple_rnn_15/transpose:y:09sequential_7/simple_rnn_15/strided_slice_2/stack:output:0;sequential_7/simple_rnn_15/strided_slice_2/stack_1:output:0;sequential_7/simple_rnn_15/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????P*
shrink_axis_mask?
Dsequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOpMsequential_7_simple_rnn_15_simple_rnn_cell_175_matmul_readvariableop_resource*
_output_shapes

:Pd*
dtype0?
5sequential_7/simple_rnn_15/simple_rnn_cell_175/MatMulMatMul3sequential_7/simple_rnn_15/strided_slice_2:output:0Lsequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Esequential_7/simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOpNsequential_7_simple_rnn_15_simple_rnn_cell_175_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0?
6sequential_7/simple_rnn_15/simple_rnn_cell_175/BiasAddBiasAdd?sequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul:product:0Msequential_7/simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
Fsequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOpOsequential_7_simple_rnn_15_simple_rnn_cell_175_matmul_1_readvariableop_resource*
_output_shapes

:dd*
dtype0?
7sequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul_1MatMul)sequential_7/simple_rnn_15/zeros:output:0Nsequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
2sequential_7/simple_rnn_15/simple_rnn_cell_175/addAddV2?sequential_7/simple_rnn_15/simple_rnn_cell_175/BiasAdd:output:0Asequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
3sequential_7/simple_rnn_15/simple_rnn_cell_175/TanhTanh6sequential_7/simple_rnn_15/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d?
8sequential_7/simple_rnn_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
*sequential_7/simple_rnn_15/TensorArrayV2_1TensorListReserveAsequential_7/simple_rnn_15/TensorArrayV2_1/element_shape:output:03sequential_7/simple_rnn_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:???a
sequential_7/simple_rnn_15/timeConst*
_output_shapes
: *
dtype0*
value	B : ~
3sequential_7/simple_rnn_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
-sequential_7/simple_rnn_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ?
 sequential_7/simple_rnn_15/whileWhile6sequential_7/simple_rnn_15/while/loop_counter:output:0<sequential_7/simple_rnn_15/while/maximum_iterations:output:0(sequential_7/simple_rnn_15/time:output:03sequential_7/simple_rnn_15/TensorArrayV2_1:handle:0)sequential_7/simple_rnn_15/zeros:output:03sequential_7/simple_rnn_15/strided_slice_1:output:0Rsequential_7/simple_rnn_15/TensorArrayUnstack/TensorListFromTensor:output_handle:0Msequential_7_simple_rnn_15_simple_rnn_cell_175_matmul_readvariableop_resourceNsequential_7_simple_rnn_15_simple_rnn_cell_175_biasadd_readvariableop_resourceOsequential_7_simple_rnn_15_simple_rnn_cell_175_matmul_1_readvariableop_resource*
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
_stateful_parallelism( *:
body2R0
.sequential_7_simple_rnn_15_while_body_11350077*:
cond2R0
.sequential_7_simple_rnn_15_while_cond_11350076*8
output_shapes'
%: : : : :?????????d: : : : : *
parallel_iterations ?
Ksequential_7/simple_rnn_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
=sequential_7/simple_rnn_15/TensorArrayV2Stack/TensorListStackTensorListStack)sequential_7/simple_rnn_15/while:output:3Tsequential_7/simple_rnn_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????d*
element_dtype0?
0sequential_7/simple_rnn_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????|
2sequential_7/simple_rnn_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: |
2sequential_7/simple_rnn_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*sequential_7/simple_rnn_15/strided_slice_3StridedSliceFsequential_7/simple_rnn_15/TensorArrayV2Stack/TensorListStack:tensor:09sequential_7/simple_rnn_15/strided_slice_3/stack:output:0;sequential_7/simple_rnn_15/strided_slice_3/stack_1:output:0;sequential_7/simple_rnn_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????d*
shrink_axis_mask?
+sequential_7/simple_rnn_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ?
&sequential_7/simple_rnn_15/transpose_1	TransposeFsequential_7/simple_rnn_15/TensorArrayV2Stack/TensorListStack:tensor:04sequential_7/simple_rnn_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????d?
 sequential_7/dropout_15/IdentityIdentity3sequential_7/simple_rnn_15/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????d?
*sequential_7/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_7_dense_7_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0?
sequential_7/dense_7/MatMulMatMul)sequential_7/dropout_15/Identity:output:02sequential_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_7/dense_7/BiasAddBiasAdd%sequential_7/dense_7/MatMul:product:03sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
IdentityIdentity%sequential_7/dense_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp+^sequential_7/dense_7/MatMul/ReadVariableOpF^sequential_7/simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOpE^sequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOpG^sequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp!^sequential_7/simple_rnn_14/whileF^sequential_7/simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOpE^sequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOpG^sequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp!^sequential_7/simple_rnn_15/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : 2Z
+sequential_7/dense_7/BiasAdd/ReadVariableOp+sequential_7/dense_7/BiasAdd/ReadVariableOp2X
*sequential_7/dense_7/MatMul/ReadVariableOp*sequential_7/dense_7/MatMul/ReadVariableOp2?
Esequential_7/simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOpEsequential_7/simple_rnn_14/simple_rnn_cell_174/BiasAdd/ReadVariableOp2?
Dsequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOpDsequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul/ReadVariableOp2?
Fsequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOpFsequential_7/simple_rnn_14/simple_rnn_cell_174/MatMul_1/ReadVariableOp2D
 sequential_7/simple_rnn_14/while sequential_7/simple_rnn_14/while2?
Esequential_7/simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOpEsequential_7/simple_rnn_15/simple_rnn_cell_175/BiasAdd/ReadVariableOp2?
Dsequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOpDsequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul/ReadVariableOp2?
Fsequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOpFsequential_7/simple_rnn_15/simple_rnn_cell_175/MatMul_1/ReadVariableOp2D
 sequential_7/simple_rnn_15/while sequential_7/simple_rnn_15/while:` \
+
_output_shapes
:?????????
-
_user_specified_namesimple_rnn_14_input
?:
?
!simple_rnn_15_while_body_113518978
4simple_rnn_15_while_simple_rnn_15_while_loop_counter>
:simple_rnn_15_while_simple_rnn_15_while_maximum_iterations#
simple_rnn_15_while_placeholder%
!simple_rnn_15_while_placeholder_1%
!simple_rnn_15_while_placeholder_27
3simple_rnn_15_while_simple_rnn_15_strided_slice_1_0s
osimple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_15_tensorarrayunstack_tensorlistfromtensor_0Z
Hsimple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource_0:PdW
Isimple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource_0:d\
Jsimple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0:dd 
simple_rnn_15_while_identity"
simple_rnn_15_while_identity_1"
simple_rnn_15_while_identity_2"
simple_rnn_15_while_identity_3"
simple_rnn_15_while_identity_45
1simple_rnn_15_while_simple_rnn_15_strided_slice_1q
msimple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_15_tensorarrayunstack_tensorlistfromtensorX
Fsimple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource:PdU
Gsimple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource:dZ
Hsimple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource:dd??>simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp?=simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp??simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp?
Esimple_rnn_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????P   ?
7simple_rnn_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemosimple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_15_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_15_while_placeholderNsimple_rnn_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????P*
element_dtype0?
=simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOpReadVariableOpHsimple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource_0*
_output_shapes

:Pd*
dtype0?
.simple_rnn_15/while/simple_rnn_cell_175/MatMulMatMul>simple_rnn_15/while/TensorArrayV2Read/TensorListGetItem:item:0Esimple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
>simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOpReadVariableOpIsimple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource_0*
_output_shapes
:d*
dtype0?
/simple_rnn_15/while/simple_rnn_cell_175/BiasAddBiasAdd8simple_rnn_15/while/simple_rnn_cell_175/MatMul:product:0Fsimple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
?simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOpReadVariableOpJsimple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0*
_output_shapes

:dd*
dtype0?
0simple_rnn_15/while/simple_rnn_cell_175/MatMul_1MatMul!simple_rnn_15_while_placeholder_2Gsimple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d?
+simple_rnn_15/while/simple_rnn_cell_175/addAddV28simple_rnn_15/while/simple_rnn_cell_175/BiasAdd:output:0:simple_rnn_15/while/simple_rnn_cell_175/MatMul_1:product:0*
T0*'
_output_shapes
:?????????d?
,simple_rnn_15/while/simple_rnn_cell_175/TanhTanh/simple_rnn_15/while/simple_rnn_cell_175/add:z:0*
T0*'
_output_shapes
:?????????d?
8simple_rnn_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!simple_rnn_15_while_placeholder_1simple_rnn_15_while_placeholder0simple_rnn_15/while/simple_rnn_cell_175/Tanh:y:0*
_output_shapes
: *
element_dtype0:???[
simple_rnn_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_15/while/addAddV2simple_rnn_15_while_placeholder"simple_rnn_15/while/add/y:output:0*
T0*
_output_shapes
: ]
simple_rnn_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :?
simple_rnn_15/while/add_1AddV24simple_rnn_15_while_simple_rnn_15_while_loop_counter$simple_rnn_15/while/add_1/y:output:0*
T0*
_output_shapes
: ?
simple_rnn_15/while/IdentityIdentitysimple_rnn_15/while/add_1:z:0^simple_rnn_15/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_15/while/Identity_1Identity:simple_rnn_15_while_simple_rnn_15_while_maximum_iterations^simple_rnn_15/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_15/while/Identity_2Identitysimple_rnn_15/while/add:z:0^simple_rnn_15/while/NoOp*
T0*
_output_shapes
: ?
simple_rnn_15/while/Identity_3IdentityHsimple_rnn_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_15/while/NoOp*
T0*
_output_shapes
: :????
simple_rnn_15/while/Identity_4Identity0simple_rnn_15/while/simple_rnn_cell_175/Tanh:y:0^simple_rnn_15/while/NoOp*
T0*'
_output_shapes
:?????????d?
simple_rnn_15/while/NoOpNoOp?^simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp>^simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp@^simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "E
simple_rnn_15_while_identity%simple_rnn_15/while/Identity:output:0"I
simple_rnn_15_while_identity_1'simple_rnn_15/while/Identity_1:output:0"I
simple_rnn_15_while_identity_2'simple_rnn_15/while/Identity_2:output:0"I
simple_rnn_15_while_identity_3'simple_rnn_15/while/Identity_3:output:0"I
simple_rnn_15_while_identity_4'simple_rnn_15/while/Identity_4:output:0"h
1simple_rnn_15_while_simple_rnn_15_strided_slice_13simple_rnn_15_while_simple_rnn_15_strided_slice_1_0"?
Gsimple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resourceIsimple_rnn_15_while_simple_rnn_cell_175_biasadd_readvariableop_resource_0"?
Hsimple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resourceJsimple_rnn_15_while_simple_rnn_cell_175_matmul_1_readvariableop_resource_0"?
Fsimple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resourceHsimple_rnn_15_while_simple_rnn_cell_175_matmul_readvariableop_resource_0"?
msimple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_15_tensorarrayunstack_tensorlistfromtensorosimple_rnn_15_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????d: : : : : 2?
>simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp>simple_rnn_15/while/simple_rnn_cell_175/BiasAdd/ReadVariableOp2~
=simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp=simple_rnn_15/while/simple_rnn_cell_175/MatMul/ReadVariableOp2?
?simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp?simple_rnn_15/while/simple_rnn_cell_175/MatMul_1/ReadVariableOp: 
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
while_cond_11350903
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11350903___redundant_placeholder06
2while_while_cond_11350903___redundant_placeholder16
2while_while_cond_11350903___redundant_placeholder26
2while_while_cond_11350903___redundant_placeholder3
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
?I
?
!__inference__traced_save_11353265
file_prefix-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopF
Bsavev2_simple_rnn_14_simple_rnn_cell_14_kernel_read_readvariableopP
Lsavev2_simple_rnn_14_simple_rnn_cell_14_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_14_simple_rnn_cell_14_bias_read_readvariableopF
Bsavev2_simple_rnn_15_simple_rnn_cell_15_kernel_read_readvariableopP
Lsavev2_simple_rnn_15_simple_rnn_cell_15_recurrent_kernel_read_readvariableopD
@savev2_simple_rnn_15_simple_rnn_cell_15_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_14_simple_rnn_cell_14_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_14_simple_rnn_cell_14_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_14_simple_rnn_cell_14_bias_m_read_readvariableopM
Isavev2_adam_simple_rnn_15_simple_rnn_cell_15_kernel_m_read_readvariableopW
Ssavev2_adam_simple_rnn_15_simple_rnn_cell_15_recurrent_kernel_m_read_readvariableopK
Gsavev2_adam_simple_rnn_15_simple_rnn_cell_15_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_14_simple_rnn_cell_14_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_14_simple_rnn_cell_14_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_14_simple_rnn_cell_14_bias_v_read_readvariableopM
Isavev2_adam_simple_rnn_15_simple_rnn_cell_15_kernel_v_read_readvariableopW
Ssavev2_adam_simple_rnn_15_simple_rnn_cell_15_recurrent_kernel_v_read_readvariableopK
Gsavev2_adam_simple_rnn_15_simple_rnn_cell_15_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopBsavev2_simple_rnn_14_simple_rnn_cell_14_kernel_read_readvariableopLsavev2_simple_rnn_14_simple_rnn_cell_14_recurrent_kernel_read_readvariableop@savev2_simple_rnn_14_simple_rnn_cell_14_bias_read_readvariableopBsavev2_simple_rnn_15_simple_rnn_cell_15_kernel_read_readvariableopLsavev2_simple_rnn_15_simple_rnn_cell_15_recurrent_kernel_read_readvariableop@savev2_simple_rnn_15_simple_rnn_cell_15_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableopIsavev2_adam_simple_rnn_14_simple_rnn_cell_14_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_14_simple_rnn_cell_14_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_14_simple_rnn_cell_14_bias_m_read_readvariableopIsavev2_adam_simple_rnn_15_simple_rnn_cell_15_kernel_m_read_readvariableopSsavev2_adam_simple_rnn_15_simple_rnn_cell_15_recurrent_kernel_m_read_readvariableopGsavev2_adam_simple_rnn_15_simple_rnn_cell_15_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopIsavev2_adam_simple_rnn_14_simple_rnn_cell_14_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_14_simple_rnn_cell_14_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_14_simple_rnn_cell_14_bias_v_read_readvariableopIsavev2_adam_simple_rnn_15_simple_rnn_cell_15_kernel_v_read_readvariableopSsavev2_adam_simple_rnn_15_simple_rnn_cell_15_recurrent_kernel_v_read_readvariableopGsavev2_adam_simple_rnn_15_simple_rnn_cell_15_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?!
?
while_body_11350370
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
$while_simple_rnn_cell_174_11350392_0:P2
$while_simple_rnn_cell_174_11350394_0:P6
$while_simple_rnn_cell_174_11350396_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
"while_simple_rnn_cell_174_11350392:P0
"while_simple_rnn_cell_174_11350394:P4
"while_simple_rnn_cell_174_11350396:PP??1while/simple_rnn_cell_174/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
1while/simple_rnn_cell_174/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2$while_simple_rnn_cell_174_11350392_0$while_simple_rnn_cell_174_11350394_0$while_simple_rnn_cell_174_11350396_0*
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
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11350318?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder:while/simple_rnn_cell_174/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity:while/simple_rnn_cell_174/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp2^while/simple_rnn_cell_174/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"J
"while_simple_rnn_cell_174_11350392$while_simple_rnn_cell_174_11350392_0"J
"while_simple_rnn_cell_174_11350394$while_simple_rnn_cell_174_11350394_0"J
"while_simple_rnn_cell_174_11350396$while_simple_rnn_cell_174_11350396_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2f
1while/simple_rnn_cell_174/StatefulPartitionedCall1while/simple_rnn_cell_174/StatefulPartitionedCall: 
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
?-
?
while_body_11352410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0L
:while_simple_rnn_cell_174_matmul_readvariableop_resource_0:PI
;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0:PN
<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0:PP
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorJ
8while_simple_rnn_cell_174_matmul_readvariableop_resource:PG
9while_simple_rnn_cell_174_biasadd_readvariableop_resource:PL
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource:PP??0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp?/while/simple_rnn_cell_174/MatMul/ReadVariableOp?1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   ?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype0?
/while/simple_rnn_cell_174/MatMul/ReadVariableOpReadVariableOp:while_simple_rnn_cell_174_matmul_readvariableop_resource_0*
_output_shapes

:P*
dtype0?
 while/simple_rnn_cell_174/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:07while/simple_rnn_cell_174/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOpReadVariableOp;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0*
_output_shapes
:P*
dtype0?
!while/simple_rnn_cell_174/BiasAddBiasAdd*while/simple_rnn_cell_174/MatMul:product:08while/simple_rnn_cell_174/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOpReadVariableOp<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0*
_output_shapes

:PP*
dtype0?
"while/simple_rnn_cell_174/MatMul_1MatMulwhile_placeholder_29while/simple_rnn_cell_174/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P?
while/simple_rnn_cell_174/addAddV2*while/simple_rnn_cell_174/BiasAdd:output:0,while/simple_rnn_cell_174/MatMul_1:product:0*
T0*'
_output_shapes
:?????????P{
while/simple_rnn_cell_174/TanhTanh!while/simple_rnn_cell_174/add:z:0*
T0*'
_output_shapes
:?????????P?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder"while/simple_rnn_cell_174/Tanh:y:0*
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
while/Identity_4Identity"while/simple_rnn_cell_174/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:?????????P?

while/NoOpNoOp1^while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0^while/simple_rnn_cell_174/MatMul/ReadVariableOp2^while/simple_rnn_cell_174/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"x
9while_simple_rnn_cell_174_biasadd_readvariableop_resource;while_simple_rnn_cell_174_biasadd_readvariableop_resource_0"z
:while_simple_rnn_cell_174_matmul_1_readvariableop_resource<while_simple_rnn_cell_174_matmul_1_readvariableop_resource_0"v
8while_simple_rnn_cell_174_matmul_readvariableop_resource:while_simple_rnn_cell_174_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :?????????P: : : : : 2d
0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp0while/simple_rnn_cell_174/BiasAdd/ReadVariableOp2b
/while/simple_rnn_cell_174/MatMul/ReadVariableOp/while/simple_rnn_cell_174/MatMul/ReadVariableOp2f
1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp1while/simple_rnn_cell_174/MatMul_1/ReadVariableOp: 
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
f
-__inference_dropout_15_layer_call_fn_11352989

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
GPU 2J 8? *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_11351051o
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
?
?
while_cond_11352409
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_16
2while_while_cond_11352409___redundant_placeholder06
2while_while_cond_11352409___redundant_placeholder16
2while_while_cond_11352409___redundant_placeholder26
2while_while_cond_11352409___redundant_placeholder3
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
:"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
W
simple_rnn_14_input@
%serving_default_simple_rnn_14_input:0?????????;
dense_70
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
/__inference_sequential_7_layer_call_fn_11351021
/__inference_sequential_7_layer_call_fn_11351502
/__inference_sequential_7_layer_call_fn_11351523
/__inference_sequential_7_layer_call_fn_11351425?
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351743
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351977
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351450
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351475?
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
#__inference__wrapped_model_11350150simple_rnn_14_input"?
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
0__inference_simple_rnn_14_layer_call_fn_11352011
0__inference_simple_rnn_14_layer_call_fn_11352022
0__inference_simple_rnn_14_layer_call_fn_11352033
0__inference_simple_rnn_14_layer_call_fn_11352044?
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
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352152
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352260
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352368
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352476?
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
-__inference_dropout_14_layer_call_fn_11352481
-__inference_dropout_14_layer_call_fn_11352486?
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
H__inference_dropout_14_layer_call_and_return_conditional_losses_11352491
H__inference_dropout_14_layer_call_and_return_conditional_losses_11352503?
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
0__inference_simple_rnn_15_layer_call_fn_11352514
0__inference_simple_rnn_15_layer_call_fn_11352525
0__inference_simple_rnn_15_layer_call_fn_11352536
0__inference_simple_rnn_15_layer_call_fn_11352547?
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
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352655
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352763
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352871
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352979?
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
-__inference_dropout_15_layer_call_fn_11352984
-__inference_dropout_15_layer_call_fn_11352989?
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
H__inference_dropout_15_layer_call_and_return_conditional_losses_11352994
H__inference_dropout_15_layer_call_and_return_conditional_losses_11353006?
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
 :d2dense_7/kernel
:2dense_7/bias
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
*__inference_dense_7_layer_call_fn_11353015?
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
E__inference_dense_7_layer_call_and_return_conditional_losses_11353025?
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
&__inference_signature_wrapper_11352000simple_rnn_14_input"?
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
9:7P2'simple_rnn_14/simple_rnn_cell_14/kernel
C:APP21simple_rnn_14/simple_rnn_cell_14/recurrent_kernel
3:1P2%simple_rnn_14/simple_rnn_cell_14/bias
9:7Pd2'simple_rnn_15/simple_rnn_cell_15/kernel
C:Add21simple_rnn_15/simple_rnn_cell_15/recurrent_kernel
3:1d2%simple_rnn_15/simple_rnn_cell_15/bias
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
6__inference_simple_rnn_cell_174_layer_call_fn_11353039
6__inference_simple_rnn_cell_174_layer_call_fn_11353053?
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
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11353070
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11353087?
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
6__inference_simple_rnn_cell_175_layer_call_fn_11353101
6__inference_simple_rnn_cell_175_layer_call_fn_11353115?
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
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11353132
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11353149?
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
%:#d2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
>:<P2.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/m
H:FPP28Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/m
8:6P2,Adam/simple_rnn_14/simple_rnn_cell_14/bias/m
>:<Pd2.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/m
H:Fdd28Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/m
8:6d2,Adam/simple_rnn_15/simple_rnn_cell_15/bias/m
%:#d2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
>:<P2.Adam/simple_rnn_14/simple_rnn_cell_14/kernel/v
H:FPP28Adam/simple_rnn_14/simple_rnn_cell_14/recurrent_kernel/v
8:6P2,Adam/simple_rnn_14/simple_rnn_cell_14/bias/v
>:<Pd2.Adam/simple_rnn_15/simple_rnn_cell_15/kernel/v
H:Fdd28Adam/simple_rnn_15/simple_rnn_cell_15/recurrent_kernel/v
8:6d2,Adam/simple_rnn_15/simple_rnn_cell_15/bias/v?
#__inference__wrapped_model_11350150ACBDFE23@?=
6?3
1?.
simple_rnn_14_input?????????
? "1?.
,
dense_7!?
dense_7??????????
E__inference_dense_7_layer_call_and_return_conditional_losses_11353025\23/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? }
*__inference_dense_7_layer_call_fn_11353015O23/?,
%?"
 ?
inputs?????????d
? "???????????
H__inference_dropout_14_layer_call_and_return_conditional_losses_11352491d7?4
-?*
$?!
inputs?????????P
p 
? ")?&
?
0?????????P
? ?
H__inference_dropout_14_layer_call_and_return_conditional_losses_11352503d7?4
-?*
$?!
inputs?????????P
p
? ")?&
?
0?????????P
? ?
-__inference_dropout_14_layer_call_fn_11352481W7?4
-?*
$?!
inputs?????????P
p 
? "??????????P?
-__inference_dropout_14_layer_call_fn_11352486W7?4
-?*
$?!
inputs?????????P
p
? "??????????P?
H__inference_dropout_15_layer_call_and_return_conditional_losses_11352994\3?0
)?&
 ?
inputs?????????d
p 
? "%?"
?
0?????????d
? ?
H__inference_dropout_15_layer_call_and_return_conditional_losses_11353006\3?0
)?&
 ?
inputs?????????d
p
? "%?"
?
0?????????d
? ?
-__inference_dropout_15_layer_call_fn_11352984O3?0
)?&
 ?
inputs?????????d
p 
? "??????????d?
-__inference_dropout_15_layer_call_fn_11352989O3?0
)?&
 ?
inputs?????????d
p
? "??????????d?
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351450{ACBDFE23H?E
>?;
1?.
simple_rnn_14_input?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351475{ACBDFE23H?E
>?;
1?.
simple_rnn_14_input?????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351743nACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_7_layer_call_and_return_conditional_losses_11351977nACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_7_layer_call_fn_11351021nACBDFE23H?E
>?;
1?.
simple_rnn_14_input?????????
p 

 
? "???????????
/__inference_sequential_7_layer_call_fn_11351425nACBDFE23H?E
>?;
1?.
simple_rnn_14_input?????????
p

 
? "???????????
/__inference_sequential_7_layer_call_fn_11351502aACBDFE23;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
/__inference_sequential_7_layer_call_fn_11351523aACBDFE23;?8
1?.
$?!
inputs?????????
p

 
? "???????????
&__inference_signature_wrapper_11352000?ACBDFE23W?T
? 
M?J
H
simple_rnn_14_input1?.
simple_rnn_14_input?????????"1?.
,
dense_7!?
dense_7??????????
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352152?ACBO?L
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
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352260?ACBO?L
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
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352368qACB??<
5?2
$?!
inputs?????????

 
p 

 
? ")?&
?
0?????????P
? ?
K__inference_simple_rnn_14_layer_call_and_return_conditional_losses_11352476qACB??<
5?2
$?!
inputs?????????

 
p

 
? ")?&
?
0?????????P
? ?
0__inference_simple_rnn_14_layer_call_fn_11352011}ACBO?L
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
0__inference_simple_rnn_14_layer_call_fn_11352022}ACBO?L
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
0__inference_simple_rnn_14_layer_call_fn_11352033dACB??<
5?2
$?!
inputs?????????

 
p 

 
? "??????????P?
0__inference_simple_rnn_14_layer_call_fn_11352044dACB??<
5?2
$?!
inputs?????????

 
p

 
? "??????????P?
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352655}DFEO?L
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
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352763}DFEO?L
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
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352871mDFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "%?"
?
0?????????d
? ?
K__inference_simple_rnn_15_layer_call_and_return_conditional_losses_11352979mDFE??<
5?2
$?!
inputs?????????P

 
p

 
? "%?"
?
0?????????d
? ?
0__inference_simple_rnn_15_layer_call_fn_11352514pDFEO?L
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
0__inference_simple_rnn_15_layer_call_fn_11352525pDFEO?L
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
0__inference_simple_rnn_15_layer_call_fn_11352536`DFE??<
5?2
$?!
inputs?????????P

 
p 

 
? "??????????d?
0__inference_simple_rnn_15_layer_call_fn_11352547`DFE??<
5?2
$?!
inputs?????????P

 
p

 
? "??????????d?
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11353070?ACB\?Y
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
Q__inference_simple_rnn_cell_174_layer_call_and_return_conditional_losses_11353087?ACB\?Y
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
6__inference_simple_rnn_cell_174_layer_call_fn_11353039?ACB\?Y
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
6__inference_simple_rnn_cell_174_layer_call_fn_11353053?ACB\?Y
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
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11353132?DFE\?Y
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
Q__inference_simple_rnn_cell_175_layer_call_and_return_conditional_losses_11353149?DFE\?Y
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
6__inference_simple_rnn_cell_175_layer_call_fn_11353101?DFE\?Y
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
6__inference_simple_rnn_cell_175_layer_call_fn_11353115?DFE\?Y
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