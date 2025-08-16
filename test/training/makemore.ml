open Base
open Ocannl
open Stdio
open Bigarray
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Asgns = Ir.Assignments

module type Backend = Ir.Backend_intf.Backend

let make_dataset names context_size =
  let aux name =
    if context_size < 1 then failwith "context size must be greater than 1";
    let n = String.length name in
    let prefix = String.make context_size '.' in
    let name = prefix ^ name ^ "." in
    List.init (n + 1) ~f:(fun i ->
        (String.sub name ~pos:i ~len:context_size, String.sub name ~pos:(i + context_size) ~len:1))
  in
  List.concat_map ~f:aux names
  |> List.map ~f:(fun (input, output) ->
         let aux s = s |> String.to_list |> List.map ~f:Datasets.Names.char_index in
         (aux input, aux output |> List.hd_exn))

(* Helper function to create one-hot tensor from integer data *)
let create_one_hot_tensor ~dimensions ~set_indices =
  (* Metal backend doesn't support double precision. *)
  let genarray = Genarray.create Bigarray.Float32 Bigarray.c_layout dimensions in
  set_indices genarray;
  TDSL.rebatch ~l:"tensor" (Ir.Ndarray.as_array Ir.Ops.Single genarray) ()

let tensor_of_int_list_list lst =
  let len = List.length lst in
  let context_size = lst |> List.hd_exn |> List.length in
  let arr = lst |> Array.of_list |> Array.map ~f:Array.of_list in
  create_one_hot_tensor ~dimensions:[| len; context_size; Datasets.Names.dict_size |]
    ~set_indices:(fun genarray ->
      (* convert to one-hot vectors *)
      for i = 0 to len - 1 do
        for j = 0 to context_size - 1 do
          Genarray.set genarray [| i; j; arr.(i).(j) |] 1.
        done
      done)

let tensor_of_int_list lst =
  let len = List.length lst in
  let arr = lst |> Array.of_list in
  create_one_hot_tensor ~dimensions:[| len; Datasets.Names.dict_size |]
    ~set_indices:(fun genarray ->
      (* convert to one-hot vectors *)
      for i = 0 to len - 1 do
        Genarray.set genarray [| i; arr.(i) |] 1.
      done)

let () =
  Utils.settings.fixed_state_for_init <- Some 13;
  Tensor.unsafe_reinitialize ();

  let names = Datasets.Names.read_names () in
  let embeddings_size = 2 in
  let context_size = 3 in
  let dataset = make_dataset names context_size in
  Stdio.printf "dataset size: %d\n%!" (List.length dataset);
  let batch_size = 1000 in
  let round_up_by = batch_size - (List.length dataset % batch_size) in
  let dataset = List.take dataset round_up_by @ dataset in

  let int_input, int_output = List.unzip dataset in
  let input_size = List.length int_input in
  Stdio.printf "input_size: %d\n%!" input_size;

  let inputs = tensor_of_int_list_list int_input in
  let outputs = tensor_of_int_list int_output in

  let n_batches = input_size / batch_size in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in

  let%op input = inputs @| batch_n in
  let%op output = outputs @| batch_n in

  (* Train.printf ~here:[%here] ~with_code:false ~with_grad:false input; *)
  let hid_dim = 100 in
  let%op mlp input =
    (* w1 is the embeddings *)
    (* For embeddings: we want to map from dict_size space to embeddings_size space
       input has shape [context_size; dict_size] where dict_size is in output position
       We need to reshape input to have dict_size in input position for multiplication *)
    let ebs =
      TDSL.param ~output_dims:[ Datasets.Names.dict_size; embeddings_size ] "embeddings" ()
    in
    (* Train.printf ~here:[%here] ~with_code:false ~with_grad:false ebs; *)
    let input_ebs = input *+ "b|ij; b|jk => b|ik" ebs in
    (* let first_layer = "w1" *+ "b|ik->g; b|ik => b|g" input_ebs *)
    let logits =
      "b2" Datasets.Names.dict_size + ("w2" * tanh ("b1" hid_dim + ("w1" * input_ebs)))
    in
    let counts = exp logits in
    counts /. (counts ++ "...|... => ...|0")
  in

  let%op output_probs = (mlp input *. output) ++ "...|... => ...|0" in
  let%op loss = neg (log output_probs) in
  let%op batch_loss = (loss ++ "...|... => 0") /. !..batch_size in

  (* When using as a tutorial, try both with the following source line included and commented out.
     Run with the option --ocannl_output_debug_files_in_build_directory=true and check the
     build_files/ directory for the generated code. *)
  Train.every_non_literal_on_host batch_loss;

  let update = Train.grad_update batch_loss in
  let%op learning_rate = 0.1 in
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  let module Backend = (val Backends.fresh_backend ()) in
  let ctx = Train.init_params (module Backend) bindings batch_loss in
  let sgd_step = Train.to_routine (module Backend) ctx bindings (Asgns.sequence [ update; sgd ]) in
  Train.printf w1 ~with_grad:false;

  let open Operation.At in
  let batch_ref = IDX.find_exn sgd_step.bindings batch_n in
  for epoch = 0 to 100 do
    for batch = 0 to n_batches - 1 do
      batch_ref := batch;
      Train.run sgd_step
    done;
    Stdio.printf "Epoch %d, loss=%f\n%!" epoch batch_loss.@[0]
  done;
  Train.printf_tree batch_loss;

  let counter_n, bindings = IDX.get_static_symbol IDX.empty in
  let%cd infer_probs = mlp "cha" in
  let%cd infer_step =
    infer_probs.forward;
    "dice" =: uniform_at !@counter_n
  in
  Train.set_on_host infer_probs.value;
  let infer_step = Train.to_routine (module Backend) sgd_step.context bindings infer_step in
  let counter_ref = IDX.find_exn infer_step.bindings counter_n in
  counter_ref := 0;

  let infer c =
    let c_one_hot = Datasets.Names.char_to_one_hot c in
    Tn.set_values cha.value c_one_hot;
    Int.incr counter_ref;
    Train.run infer_step;
    let dice_value = dice.@[0] in

    let rec aux i sum =
      let prob = infer_probs.@{[| i |]} in
      let new_sum = sum +. prob in
      if Float.compare new_sum dice_value > 0 then List.nth_exn Datasets.Names.letters_with_dot i
      else aux (i + 1) new_sum
    in

    aux 0 0.
  in

  let gen_name () =
    let rec aux c name =
      if (Char.equal c '.' || Char.equal c ' ') && not (String.equal name "") then name
      else
        let next_char = infer c in
        aux next_char (name ^ String.make 1 c)
    in
    let name_with_dot = aux '.' "" in
    String.drop_prefix name_with_dot 1
  in

  let names = Array.init 20 ~f:(fun _ -> gen_name ()) in
  Array.iter names ~f:print_endline
