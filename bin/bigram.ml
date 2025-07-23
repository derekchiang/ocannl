open Base
open Ocannl
open Stdio
open Bigarray
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Rand = Ir.Rand.Lib
module Asgns = Ir.Assignments

module type Backend = Ir.Backend_intf.Backend

let read_names () = In_channel.read_lines "bin/names.txt"

let bigrams s =
  let chars = String.to_list s in
  let front = '.' :: chars in
  let back = chars @ [ '.' ] in
  List.zip_exn front back

let get_all_bigrams () = List.(read_names () >>| bigrams |> concat)
let letters = List.init 26 ~f:(fun i -> Char.of_int_exn (Char.to_int 'a' + i))
let letters_with_dot = '.' :: letters

let char_to_index_tbl =
  let tbl = Hashtbl.create (module Char) in
  List.iteri letters_with_dot ~f:(fun i c -> Hashtbl.set tbl ~key:c ~data:i);
  tbl

let char_index c =
  match Hashtbl.find char_to_index_tbl c with
  | Some i -> i
  | None -> failwith (Printf.sprintf "Character not found: %c" c)

let bigrams_to_indices bigrams = List.(bigrams >>| fun (c1, c2) -> (char_index c1, char_index c2))
let print_tensor t = Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default t

let _one_hot ~num_classes indices =
  let num_classes = num_classes - 1 in
  let%op classes = TDSL.range num_classes in
  let%op indices_expanded = indices ++ "b|1 => b|i" in
  let%op classes_expanded = classes ++ "i => b|i" in
  let%op one_hot = indices_expanded = classes_expanded in
  one_hot

let char_to_one_hot c =
  let c_index = char_index c in
  let arr = Array.create ~len:27 0. in
  arr.(c_index) <- 1.;
  arr

let tensor_of_int_list lst =
  let len = List.length lst in
  let arr = lst |> List.map ~f:Float.of_int |> Array.of_list in
  let genarray = Genarray.create Bigarray.Float64 Bigarray.c_layout [| len; 27 |] in
  (* convert to one-hot vectors *)
  for i = 0 to len - 1 do
    Genarray.set genarray [| i; Int.of_float arr.(i) |] 1.
  done;
  let tensor = TDSL.rebatch ~l:"tensor" (Ir.Ndarray.as_array Ir.Ops.Double genarray) in
  print_tensor tensor;
  tensor

let () =
  let seed = 13 in
  Rand.init seed;
  Utils.settings.fixed_state_for_init <- Some seed;

  let bigrams = get_all_bigrams () |> bigrams_to_indices in
  let input_size = 100 in

  let int_input, int_output = List.unzip (List.take bigrams input_size) in

  let inputs = tensor_of_int_list int_input in
  let outputs = tensor_of_int_list int_output in

  (* let inputs = input_tensor |> one_hot ~num_classes:27 in let outputs = output_tensor |> one_hot
     ~num_classes:27 in Train.set_hosted inputs.value; *)
  let batch_size = 100 in
  let n_batches = input_size / batch_size in
  let batch_n, bindings = IDX.get_static_symbol ~static_range:n_batches IDX.empty in

  let%op input = inputs @| batch_n in
  let%op output = outputs @| batch_n in
  (* let%cd _ = input =: 0 ++ "i=>32|i" in let%cd _ = output =: 0 ++ "i=>32|i" in *)

  let mlp input =
    let random_weights = Array.init 27 ~f:(fun _ -> Random.float 2.0 -. 1.0) in
    (* let w = TDSL.param ~values:random_weights ~output_dims:[ 27 ] "w" in *)
    let%op logits = "w" 27 *. input in
    Tn.set_values w.value random_weights;
    Train.set_hosted logits.value;

    let%op counts = exp logits in
    Train.set_hosted counts.value;

    let%op probs = counts /. (counts ++ "b|... => b|0") in
    Train.set_hosted probs.value;

    probs
  in

  let%op output_probs = (mlp input *. output) ++ "b|... => b|0" in
  Train.set_hosted output_probs.value;

  let%op loss = neg (log output_probs) in
  Train.set_hosted loss.value;

  let%op batch_loss = (loss ++ "...|... => 0") /. !..batch_size in
  Train.set_hosted batch_loss.value;

  let update = Train.grad_update batch_loss in
  let%op learning_rate = 1 in
  let sgd = Train.sgd_update ~learning_rate batch_loss in

  let module Backend = (val Backends.fresh_backend ()) in
  let ctx = Train.init_params (module Backend) bindings batch_loss in
  let routine = Train.to_routine (module Backend) ctx bindings (Asgns.sequence [ update; sgd ]) in

  let open Operation.At in
  let batch_ref = IDX.find_exn routine.bindings batch_n in
  (* running the init sets the weights to zero... how to avoid that? *)
  (* Train.run init; *)
  for epoch = 0 to 100 do
    for batch = 0 to n_batches - 1 do
      batch_ref := batch;
      Train.run routine
      (* Train.forward_and_forget (module Backend) ctx batch_loss; *)
      (* print_tensor inputs;
    print_tensor logits;
    print_tensor counts;
    print_tensor probs;
    print_tensor output_probs;
    print_tensor loss;
    print_tensor batch_loss *)
    done;
    Stdio.printf "Epoch %d, loss=%f\n%!" epoch batch_loss.@[0]
  done;
  print_tensor inputs;
  print_tensor input;
  print_tensor loss;
  print_tensor batch_loss;

  let%cd infer_probs = mlp "cha" in
  Train.set_on_host infer_probs.value;
  let infer_probs_routine =
    Train.to_routine
      (module Backend)
      routine.context IDX.empty
      [%cd
        ~~("probs infer";
           infer_probs.forward)]
  in
  let infer c =
    let c_one_hot = char_to_one_hot c in
    Tn.set_values cha.value c_one_hot;
    Utils.capture_stdout_logs @@ fun () ->
    Train.run infer_probs_routine;
    infer_probs.@[char_index c]
  in
  Stdio.printf "Prob: %f\n" (infer 'c')
